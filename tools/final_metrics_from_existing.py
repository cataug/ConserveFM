from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import tools.eval_conservefm as ev
from conservefm.corruptions import PhysicsCorruptor
from conservefm.physics import compute_physics_losses


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path.home() / "ConserveFM"

OUT = ROOT / "reports" / "final_metrics_s42"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 32
STRESS_N = 64

LEADS = [6, 24, 72]

STRESS_TYPES = [
    "moisture",
    "hydrostatic",
    "advection",
    "spectral",
    "range_extreme",
]

SEVERITIES = [0.25, 0.5, 1.0, 2.0]

PHYSICS = [
    "moisture",
    "hydrostatic",
    "advection",
    "spectral",
]

MODELS = {
    "OLD_Full": {
        "method": "conservefm_full",
        "checkpoint_dir": ROOT / "runs/train_climax_conservefm_full_s42",
    },
    "No_Pretrain": {
        "method": "no_corruption_pretrain",
        "checkpoint_dir": ROOT / "runs/train_climax_no_corruption_pretrain_s42",
    },
    "Direct_State": {
        "method": "direct_state_prediction",
        "checkpoint_dir": ROOT / "runs/train_climax_direct_state_prediction_s42",
    },
    "Smart_Final": {
        "method": "conservefm_full",
        "checkpoint_dir": ROOT / "runs/train_climax_conservefm_smartfinal_s42",
    },
}


# ======================================================================
# DATA
# ======================================================================

reader = ev.ERA5Reader(
    ROOT / "data/ERA5_WeatherBench2_1979_2022"
)

normalizer = ev.Normalizer.load(
    ROOT / "manifests/research_v1/era5_channel_stats.npz"
)

provider, provider_mode = ev.make_forecast_provider(
    "climax",
    reader,
    ROOT / "forecasts",
    smoke=False,
)

test_rows = ev.read_examples(
    ROOT / "manifests/research_v1/era5_examples.csv",
    "test",
)

latitude = torch.as_tensor(
    reader.latitude,
    dtype=torch.float32,
    device=DEVICE,
)

N_CHANNELS = reader.layout.n_channels


# ======================================================================
# CHANNEL NAMES
# ======================================================================

def find_channel_names(obj, n):
    # Direct attributes first.
    for attr in (
        "channel_names",
        "channels",
        "names",
        "expanded_channels",
    ):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if (
                isinstance(v, (list, tuple))
                and len(v) == n
                and all(isinstance(x, str) for x in v)
            ):
                return list(v)

    # Search layout.to_dict() recursively.
    try:
        d = obj.to_dict()
    except Exception:
        d = {}

    def rec(x):
        if isinstance(x, dict):
            for v in x.values():
                z = rec(v)
                if z is not None:
                    return z

        if isinstance(x, (list, tuple)):
            if (
                len(x) == n
                and all(isinstance(v, str) for v in x)
            ):
                return list(x)

            for v in x:
                z = rec(v)
                if z is not None:
                    return z

        return None

    found = rec(d)

    if found is not None:
        return found

    return [f"channel_{i:02d}" for i in range(n)]


CHANNEL_NAMES = find_channel_names(
    reader.layout,
    N_CHANNELS,
)

print(f"[SETUP] device={DEVICE}")
print(f"[SETUP] channels={N_CHANNELS}")
print(f"[SETUP] provider={provider_mode}")
print(f"[SETUP] test rows={len(test_rows)}")
print("[SETUP] first channels:", CHANNEL_NAMES[:10])


# ======================================================================
# HELPERS
# ======================================================================

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def load_batch(rows, lead):
    previous = torch.stack([
        torch.from_numpy(
            reader.read_state(
                int(r["context_end_idx"])
            )
        )
        for r in rows
    ]).to(
        DEVICE,
        non_blocking=True,
    )

    target = torch.stack([
        torch.from_numpy(
            reader.read_state(
                int(r["target_idx"])
            )
        )
        for r in rows
    ]).to(
        DEVICE,
        non_blocking=True,
    )

    raw = torch.stack([
        torch.from_numpy(
            provider.get(
                int(r["context_end_idx"]),
                int(lead),
            )
        )
        for r in rows
    ]).to(
        DEVICE,
        non_blocking=True,
    )

    return previous, raw, target


def spatial_weights(x):
    """
    Cos(latitude) area weights with shape [1, 1, H, W].
    Works regardless of whether latitude is the last or penultimate axis.
    """
    nlat = len(reader.latitude)

    wlat = torch.cos(
        latitude * math.pi / 180.0
    ).clamp_min(0.0)

    h, w = x.shape[-2], x.shape[-1]

    if h == nlat:
        sw = wlat[:, None].expand(h, w)

    elif w == nlat:
        sw = wlat[None, :].expand(h, w)

    else:
        raise RuntimeError(
            f"Cannot locate latitude dimension: "
            f"tensor spatial={x.shape[-2:]}, nlat={nlat}"
        )

    return sw[None, None, :, :]


class MetricAccumulator:
    """
    Accumulates area-weighted metrics.

    Normalized metrics use the EXISTING training-set normalizer:
      z = (x - train_mean) / train_std

    Hence normalized_rmse is RMSE measured in training-standard-deviation
    units, with all channels placed on a comparable scale.
    """

    def __init__(self):
        c = N_CHANNELS

        self.den = 0.0

        self.sse_phys = torch.zeros(c, dtype=torch.float64)
        self.sae_phys = torch.zeros(c, dtype=torch.float64)

        self.sse_norm = torch.zeros(c, dtype=torch.float64)
        self.sae_norm = torch.zeros(c, dtype=torch.float64)

        self.xy = torch.zeros(c, dtype=torch.float64)
        self.xx = torch.zeros(c, dtype=torch.float64)
        self.yy = torch.zeros(c, dtype=torch.float64)

        self.n_examples = 0

    def add(self, pred_phys, target_phys):
        pred_phys = pred_phys.float()
        target_phys = target_phys.float()

        b = pred_phys.shape[0]

        w = spatial_weights(pred_phys)

        spatial_den = float(w.sum().detach().cpu())
        self.den += b * spatial_den
        self.n_examples += b

        ep = pred_phys - target_phys

        self.sse_phys += (
            (ep.square() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        self.sae_phys += (
            (ep.abs() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        pred_norm = normalizer.normalize_torch(
            pred_phys
        ).float()

        target_norm = normalizer.normalize_torch(
            target_phys
        ).float()

        en = pred_norm - target_norm

        self.sse_norm += (
            (en.square() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        self.sae_norm += (
            (en.abs() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        # Anomaly correlation relative to TRAIN mean.
        # normalize_torch already subtracts training mean.
        self.xy += (
            (pred_norm * target_norm * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        self.xx += (
            (pred_norm.square() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

        self.yy += (
            (target_norm.square() * w)
            .sum(dim=(0, 2, 3))
            .double()
            .detach()
            .cpu()
        )

    def channel_arrays(self):
        den = max(self.den, 1e-12)

        rmse_phys = torch.sqrt(
            self.sse_phys / den
        ).numpy()

        mae_phys = (
            self.sae_phys / den
        ).numpy()

        nrmse = torch.sqrt(
            self.sse_norm / den
        ).numpy()

        nmae = (
            self.sae_norm / den
        ).numpy()

        acc = (
            self.xy
            / torch.sqrt(
                self.xx.clamp_min(1e-30)
                * self.yy.clamp_min(1e-30)
            )
        ).numpy()

        return {
            "rmse_phys": rmse_phys,
            "mae_phys": mae_phys,
            "nrmse": nrmse,
            "nmae": nmae,
            "acc": acc,
        }

    def summary(self):
        c = N_CHANNELS
        den = max(self.den, 1e-12)

        arr = self.channel_arrays()

        return {
            # All channels standardized first, then pooled.
            "nrmse": float(
                math.sqrt(
                    float(self.sse_norm.sum())
                    / (den * c)
                )
            ),
            "nmae": float(
                float(self.sae_norm.sum())
                / (den * c)
            ),

            # Mean per-channel area-weighted anomaly correlation.
            "acc": float(
                np.nanmean(arr["acc"])
            ),

            # Kept ONLY for continuity with old diagnostic metric.
            # Units are mixed across channels.
            "rmse_phys_mixed": float(
                math.sqrt(
                    float(self.sse_phys.sum())
                    / (den * c)
                )
            ),
            "mae_phys_mixed": float(
                float(self.sae_phys.sum())
                / (den * c)
            ),

            "n_examples": int(self.n_examples),
        }


class PhysicsAccumulator:
    def __init__(self):
        self.sum = {
            k: 0.0
            for k in PHYSICS
        }
        self.n = 0

    def add(self, state, previous, lead):
        b = state.shape[0]

        lead_tensor = torch.full(
            (b,),
            float(lead),
            device=DEVICE,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            d = compute_physics_losses(
                state.float(),
                previous.float(),
                reader.layout,
                lead_tensor,
                PHYSICS,
                latitude_deg=latitude,
            )

        for k in PHYSICS:
            self.sum[k] += (
                float(d[k].detach().cpu()) * b
            )

        self.n += b

    def mean(self):
        n = max(self.n, 1)

        return {
            k: self.sum[k] / n
            for k in PHYSICS
        }


def metric_columns(prefix, acc):
    s = acc.summary()

    return {
        f"{prefix}_nrmse": s["nrmse"],
        f"{prefix}_nmae": s["nmae"],
        f"{prefix}_acc": s["acc"],
        f"{prefix}_rmse_phys_mixed": s["rmse_phys_mixed"],
        f"{prefix}_mae_phys_mixed": s["mae_phys_mixed"],
    }


def physics_columns(prefix, phys_acc):
    d = phys_acc.mean()

    return {
        f"{prefix}_phys_{k}": d[k]
        for k in PHYSICS
    }


def physics_improvement_columns(
    before,
    after,
):
    b = before.mean()
    a = after.mean()

    out = {}

    for k in PHYSICS:
        out[f"physics_improvement_{k}_pct"] = (
            100.0 * (b[k] - a[k])
            / max(abs(b[k]), 1e-20)
        )

    return out


# ======================================================================
# FIX STRESS SAMPLE SET ONCE
# ======================================================================

stress_rows = {}

for lead in LEADS:
    stress_rows[lead] = ev.sample_rows_for_stress(
        test_rows,
        lead,
        STRESS_N,
        12345,
    )

    print(
        f"[STRESS SET] lead={lead} "
        f"n={len(stress_rows[lead])}"
    )


# ======================================================================
# OUTPUT CONTAINERS
# ======================================================================

clean_summary_rows = []
clean_channel_rows = []

stress_condition_rows = []
stress_channel_rows = []


# ======================================================================
# MODEL LOOP
# ======================================================================

for model_name, spec in MODELS.items():

    print("\n" + "=" * 110)
    print("MODEL:", model_name)
    print("=" * 110)

    args = SimpleNamespace(
        method=spec["method"],
        checkpoint_dir=spec["checkpoint_dir"],
        backbone="climax",
        family="main",
        forecast_root=ROOT / "forecasts",
        smoke=False,
    )

    model, cfg = ev.load_model(
        args,
        reader,
        DEVICE,
    )

    model.eval()

    # ------------------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------------------

    clean_rep_all = MetricAccumulator()
    clean_raw_all = MetricAccumulator()

    clean_rep_phys_all = PhysicsAccumulator()
    clean_raw_phys_all = PhysicsAccumulator()
    clean_target_phys_all = PhysicsAccumulator()

    for lead in LEADS:

        lead_rows = [
            r for r in test_rows
            if int(r["lead_hours"]) == lead
        ]

        rep_acc = MetricAccumulator()
        raw_acc = MetricAccumulator()

        rep_phys = PhysicsAccumulator()
        raw_phys = PhysicsAccumulator()
        target_phys = PhysicsAccumulator()

        print(
            f"[CLEAN] {model_name} lead={lead} "
            f"n={len(lead_rows)}"
        )

        done = 0

        for rr in chunks(
            lead_rows,
            BATCH,
        ):
            previous, raw, target = load_batch(
                rr,
                lead,
            )

            with torch.inference_mode():
                repaired, out = ev.predict(
                    raw,
                    previous,
                    lead,
                    model,
                    cfg,
                    spec["method"],
                    normalizer,
                    reader.layout,
                    DEVICE,
                )

            rep_acc.add(
                repaired,
                target,
            )

            raw_acc.add(
                raw,
                target,
            )

            clean_rep_all.add(
                repaired,
                target,
            )

            clean_raw_all.add(
                raw,
                target,
            )

            rep_phys.add(
                repaired,
                previous,
                lead,
            )

            raw_phys.add(
                raw,
                previous,
                lead,
            )

            target_phys.add(
                target,
                previous,
                lead,
            )

            clean_rep_phys_all.add(
                repaired,
                previous,
                lead,
            )

            clean_raw_phys_all.add(
                raw,
                previous,
                lead,
            )

            clean_target_phys_all.add(
                target,
                previous,
                lead,
            )

            done += len(rr)

            if (
                done % 1024 < len(rr)
                or done == len(lead_rows)
            ):
                print(
                    f"[CLEAN] {model_name} "
                    f"lead={lead} "
                    f"{done}/{len(lead_rows)}",
                    flush=True,
                )

        row = {
            "model": model_name,
            "lead_hours": lead,
            "n_examples": rep_acc.n_examples,
        }

        row.update(
            metric_columns(
                "repaired",
                rep_acc,
            )
        )

        row.update(
            metric_columns(
                "raw_forecast",
                raw_acc,
            )
        )

        row.update(
            physics_columns(
                "repaired",
                rep_phys,
            )
        )

        row.update(
            physics_columns(
                "raw_forecast",
                raw_phys,
            )
        )

        row.update(
            physics_columns(
                "era5_target",
                target_phys,
            )
        )

        row.update(
            physics_improvement_columns(
                raw_phys,
                rep_phys,
            )
        )

        row["nrmse_improvement_pct"] = (
            100.0
            * (
                row["raw_forecast_nrmse"]
                - row["repaired_nrmse"]
            )
            / max(
                row["raw_forecast_nrmse"],
                1e-20,
            )
        )

        clean_summary_rows.append(row)

        rch = rep_acc.channel_arrays()
        xch = raw_acc.channel_arrays()

        for ci, channel in enumerate(
            CHANNEL_NAMES
        ):
            clean_channel_rows.append({
                "model": model_name,
                "lead_hours": lead,
                "channel_index": ci,
                "channel": channel,

                "repaired_nrmse":
                    float(rch["nrmse"][ci]),

                "raw_forecast_nrmse":
                    float(xch["nrmse"][ci]),

                "repaired_nmae":
                    float(rch["nmae"][ci]),

                "raw_forecast_nmae":
                    float(xch["nmae"][ci]),

                "repaired_acc":
                    float(rch["acc"][ci]),

                "raw_forecast_acc":
                    float(xch["acc"][ci]),

                "repaired_rmse_phys":
                    float(rch["rmse_phys"][ci]),

                "raw_forecast_rmse_phys":
                    float(xch["rmse_phys"][ci]),

                "repaired_mae_phys":
                    float(rch["mae_phys"][ci]),

                "raw_forecast_mae_phys":
                    float(xch["mae_phys"][ci]),
            })

    # ALL leads clean summary.
    row = {
        "model": model_name,
        "lead_hours": "ALL",
        "n_examples":
            clean_rep_all.n_examples,
    }

    row.update(
        metric_columns(
            "repaired",
            clean_rep_all,
        )
    )

    row.update(
        metric_columns(
            "raw_forecast",
            clean_raw_all,
        )
    )

    row.update(
        physics_columns(
            "repaired",
            clean_rep_phys_all,
        )
    )

    row.update(
        physics_columns(
            "raw_forecast",
            clean_raw_phys_all,
        )
    )

    row.update(
        physics_columns(
            "era5_target",
            clean_target_phys_all,
        )
    )

    row.update(
        physics_improvement_columns(
            clean_raw_phys_all,
            clean_rep_phys_all,
        )
    )

    row["nrmse_improvement_pct"] = (
        100.0
        * (
            row["raw_forecast_nrmse"]
            - row["repaired_nrmse"]
        )
        / max(
            row["raw_forecast_nrmse"],
            1e-20,
        )
    )

    clean_summary_rows.append(row)

    # ------------------------------------------------------------------
    # STRESS
    # ------------------------------------------------------------------

    # Re-create per model with identical seed.
    # Therefore every model receives the same stochastic corruptions.
    corruptor = PhysicsCorruptor(
        reader.layout,
        STRESS_TYPES,
        seed=42,
    )

    for lead in LEADS:

        selected = stress_rows[lead]

        for corruption in STRESS_TYPES:

            for severity in SEVERITIES:

                input_acc = MetricAccumulator()
                repaired_acc = MetricAccumulator()

                input_phys = PhysicsAccumulator()
                repaired_phys = PhysicsAccumulator()
                target_phys = PhysicsAccumulator()

                loc_sum = 0.0
                loc_n = 0

                type_correct = 0
                type_n = 0

                repair_sum = 0.0
                repair_n = 0

                router_sum = {
                    k: 0.0
                    for k in cfg.enabled_constraints
                }
                router_n = 0

                for rr in chunks(
                    selected,
                    BATCH,
                ):
                    previous, raw, target = load_batch(
                        rr,
                        lead,
                    )

                    with torch.inference_mode():
                        corr = corruptor.corrupt(
                            raw,
                            severity=severity,
                            force_type=corruption,
                        )

                        repaired, out = ev.predict(
                            corr.state,
                            previous,
                            lead,
                            model,
                            cfg,
                            spec["method"],
                            normalizer,
                            reader.layout,
                            DEVICE,
                        )

                    # Forecast metrics.
                    input_acc.add(
                        corr.state,
                        target,
                    )

                    repaired_acc.add(
                        repaired,
                        target,
                    )

                    # Physics metrics.
                    input_phys.add(
                        corr.state,
                        previous,
                        lead,
                    )

                    repaired_phys.add(
                        repaired,
                        previous,
                        lead,
                    )

                    target_phys.add(
                        target,
                        previous,
                        lead,
                    )

                    # Localization.
                    if (
                        getattr(
                            out,
                            "localization_logits",
                            None,
                        )
                        is not None
                    ):
                        pred_mask = (
                            torch.sigmoid(
                                out.localization_logits
                            ) >= 0.5
                        )

                        true_mask = (
                            corr.mask >= 0.5
                        )

                        pf = pred_mask.flatten(1)
                        tf = true_mask.flatten(1)

                        inter = (
                            pf & tf
                        ).float().sum(dim=1)

                        union = (
                            pf | tf
                        ).float().sum(dim=1)

                        valid = union > 0

                        if valid.any():
                            vals = (
                                inter[valid]
                                / union[valid]
                            )

                            loc_sum += float(
                                vals.sum().cpu()
                            )

                            loc_n += int(
                                vals.numel()
                            )

                    # Corruption type.
                    if (
                        getattr(
                            out,
                            "type_logits",
                            None,
                        )
                        is not None
                    ):
                        pred_type = (
                            out.type_logits
                            .argmax(dim=1)
                        )

                        true_type = (
                            corr.type_index
                            .to(pred_type.device)
                        )

                        type_correct += int(
                            (
                                pred_type
                                == true_type
                            )
                            .sum()
                            .cpu()
                        )

                        type_n += int(
                            pred_type.numel()
                        )

                    # Repair magnitude in normalized model space.
                    delta = getattr(
                        out,
                        "delta_norm",
                        None,
                    )

                    if delta is not None:
                        per_example = (
                            delta.float()
                            .abs()
                            .flatten(1)
                            .mean(dim=1)
                        )
                    else:
                        repaired_norm = (
                            normalizer
                            .normalize_torch(
                                repaired.float()
                            )
                        )

                        input_norm = (
                            normalizer
                            .normalize_torch(
                                corr.state.float()
                            )
                        )

                        per_example = (
                            (
                                repaired_norm
                                - input_norm
                            )
                            .abs()
                            .flatten(1)
                            .mean(dim=1)
                        )

                    repair_sum += float(
                        per_example.sum().cpu()
                    )

                    repair_n += int(
                        per_example.numel()
                    )

                    # Router.
                    rw = getattr(
                        out,
                        "router_weights",
                        None,
                    )

                    if rw is not None:
                        for j, name in enumerate(
                            cfg.enabled_constraints
                        ):
                            router_sum[name] += float(
                                rw[:, j]
                                .float()
                                .sum()
                                .cpu()
                            )

                        router_n += int(
                            rw.shape[0]
                        )

                row = {
                    "model": model_name,
                    "lead_hours": lead,
                    "corruption": corruption,
                    "severity": severity,
                    "n_examples":
                        repaired_acc.n_examples,

                    "localization_iou":
                        (
                            loc_sum / loc_n
                            if loc_n
                            else np.nan
                        ),

                    "type_accuracy":
                        (
                            type_correct / type_n
                            if type_n
                            else np.nan
                        ),

                    "repair_l1_norm":
                        (
                            repair_sum / repair_n
                            if repair_n
                            else np.nan
                        ),
                }

                row.update(
                    metric_columns(
                        "repaired",
                        repaired_acc,
                    )
                )

                row.update(
                    metric_columns(
                        "corrupted_input",
                        input_acc,
                    )
                )

                row.update(
                    physics_columns(
                        "repaired",
                        repaired_phys,
                    )
                )

                row.update(
                    physics_columns(
                        "corrupted_input",
                        input_phys,
                    )
                )

                row.update(
                    physics_columns(
                        "era5_target",
                        target_phys,
                    )
                )

                row.update(
                    physics_improvement_columns(
                        input_phys,
                        repaired_phys,
                    )
                )

                row["nrmse_improvement_pct"] = (
                    100.0
                    * (
                        row[
                            "corrupted_input_nrmse"
                        ]
                        - row[
                            "repaired_nrmse"
                        ]
                    )
                    / max(
                        row[
                            "corrupted_input_nrmse"
                        ],
                        1e-20,
                    )
                )

                for name in cfg.enabled_constraints:
                    row[
                        f"router_{name}"
                    ] = (
                        router_sum[name]
                        / router_n
                        if router_n
                        else np.nan
                    )

                stress_condition_rows.append(
                    row
                )

                rch = (
                    repaired_acc
                    .channel_arrays()
                )

                ich = (
                    input_acc
                    .channel_arrays()
                )

                for ci, channel in enumerate(
                    CHANNEL_NAMES
                ):
                    stress_channel_rows.append({
                        "model":
                            model_name,
                        "lead_hours":
                            lead,
                        "corruption":
                            corruption,
                        "severity":
                            severity,
                        "channel_index":
                            ci,
                        "channel":
                            channel,

                        "repaired_nrmse":
                            float(
                                rch[
                                    "nrmse"
                                ][ci]
                            ),

                        "corrupted_input_nrmse":
                            float(
                                ich[
                                    "nrmse"
                                ][ci]
                            ),

                        "repaired_nmae":
                            float(
                                rch[
                                    "nmae"
                                ][ci]
                            ),

                        "corrupted_input_nmae":
                            float(
                                ich[
                                    "nmae"
                                ][ci]
                            ),

                        "repaired_acc":
                            float(
                                rch[
                                    "acc"
                                ][ci]
                            ),

                        "corrupted_input_acc":
                            float(
                                ich[
                                    "acc"
                                ][ci]
                            ),

                        "repaired_rmse_phys":
                            float(
                                rch[
                                    "rmse_phys"
                                ][ci]
                            ),

                        "corrupted_input_rmse_phys":
                            float(
                                ich[
                                    "rmse_phys"
                                ][ci]
                            ),
                    })

                print(
                    f"[STRESS] "
                    f"{model_name:12s} "
                    f"lead={lead:2d} "
                    f"{corruption:14s} "
                    f"sev={severity:<4} "
                    f"NRMSE="
                    f"{row['repaired_nrmse']:.4f} "
                    f"type="
                    f"{row['type_accuracy']:.3f} "
                    f"iou="
                    f"{row['localization_iou']:.3f}",
                    flush=True,
                )


# ======================================================================
# SAVE RAW TABLES
# ======================================================================

clean_df = pd.DataFrame(
    clean_summary_rows
)

clean_ch_df = pd.DataFrame(
    clean_channel_rows
)

stress_df = pd.DataFrame(
    stress_condition_rows
)

stress_ch_df = pd.DataFrame(
    stress_channel_rows
)

clean_df.to_csv(
    OUT / "clean_summary.csv",
    index=False,
)

clean_ch_df.to_csv(
    OUT / "clean_by_channel.csv",
    index=False,
)

stress_df.to_csv(
    OUT / "stress_conditions.csv",
    index=False,
)

stress_ch_df.to_csv(
    OUT / "stress_by_channel_conditions.csv",
    index=False,
)


# ======================================================================
# AGGREGATES
# ======================================================================

# Macro = every lead × corruption × severity condition receives equal weight.
macro_cols = [
    "repaired_nrmse",
    "repaired_nmae",
    "repaired_acc",
    "repaired_rmse_phys_mixed",
    "repaired_mae_phys_mixed",
    "corrupted_input_nrmse",
    "nrmse_improvement_pct",
    "localization_iou",
    "type_accuracy",
    "repair_l1_norm",
]

for p in PHYSICS:
    macro_cols += [
        f"repaired_phys_{p}",
        f"corrupted_input_phys_{p}",
        f"era5_target_phys_{p}",
        f"physics_improvement_{p}_pct",
    ]

macro_cols = [
    c for c in macro_cols
    if c in stress_df.columns
]

stress_macro = (
    stress_df
    .groupby("model")[macro_cols]
    .mean()
    .reset_index()
)

stress_macro.to_csv(
    OUT / "stress_macro_summary.csv",
    index=False,
)

stress_by_corruption = (
    stress_df
    .groupby(
        ["model", "corruption"]
    )[macro_cols]
    .mean()
    .reset_index()
)

stress_by_corruption.to_csv(
    OUT / "stress_by_corruption.csv",
    index=False,
)

stress_by_severity = (
    stress_df
    .groupby(
        ["model", "severity"]
    )[macro_cols]
    .mean()
    .reset_index()
)

stress_by_severity.to_csv(
    OUT / "stress_by_severity.csv",
    index=False,
)

stress_by_lead = (
    stress_df
    .groupby(
        ["model", "lead_hours"]
    )[macro_cols]
    .mean()
    .reset_index()
)

stress_by_lead.to_csv(
    OUT / "stress_by_lead.csv",
    index=False,
)

stress_channel_macro = (
    stress_ch_df
    .groupby(
        [
            "model",
            "channel_index",
            "channel",
        ]
    )[
        [
            "repaired_nrmse",
            "corrupted_input_nrmse",
            "repaired_nmae",
            "corrupted_input_nmae",
            "repaired_acc",
            "corrupted_input_acc",
            "repaired_rmse_phys",
            "corrupted_input_rmse_phys",
        ]
    ]
    .mean()
    .reset_index()
)

stress_channel_macro.to_csv(
    OUT / "stress_by_channel_macro.csv",
    index=False,
)


# ======================================================================
# FINAL ONE-ROW-PER-MODEL TABLE
# ======================================================================

clean_all = clean_df[
    clean_df["lead_hours"].astype(str) == "ALL"
].copy()

clean_all = clean_all.rename(
    columns={
        "repaired_nrmse":
            "clean_nrmse",
        "repaired_nmae":
            "clean_nmae",
        "repaired_acc":
            "clean_acc",
        "repaired_rmse_phys_mixed":
            "clean_rmse_phys_mixed",
        "raw_forecast_nrmse":
            "clean_raw_nrmse",
        "nrmse_improvement_pct":
            "clean_nrmse_improvement_pct",
    }
)

final = clean_all[
    [
        "model",
        "clean_nrmse",
        "clean_nmae",
        "clean_acc",
        "clean_rmse_phys_mixed",
        "clean_raw_nrmse",
        "clean_nrmse_improvement_pct",
    ]
].merge(
    stress_macro[
        [
            "model",
            "repaired_nrmse",
            "repaired_nmae",
            "repaired_acc",
            "repaired_rmse_phys_mixed",
            "corrupted_input_nrmse",
            "nrmse_improvement_pct",
            "localization_iou",
            "type_accuracy",
            "repair_l1_norm",
        ]
    ],
    on="model",
    how="left",
)

final = final.rename(
    columns={
        "repaired_nrmse":
            "macro_stress_nrmse",
        "repaired_nmae":
            "macro_stress_nmae",
        "repaired_acc":
            "macro_stress_acc",
        "repaired_rmse_phys_mixed":
            "macro_stress_rmse_phys_mixed",
        "corrupted_input_nrmse":
            "macro_corrupted_input_nrmse",
        "nrmse_improvement_pct":
            "macro_stress_nrmse_improvement_pct",
        "localization_iou":
            "macro_localization_iou",
        "type_accuracy":
            "macro_type_accuracy",
        "repair_l1_norm":
            "macro_repair_l1_norm",
    }
)

final.to_csv(
    OUT / "FINAL_SUMMARY.csv",
    index=False,
)


# ======================================================================
# PROTOCOL METADATA
# ======================================================================

meta = {
    "backbone": "climax",
    "seed": 42,
    "test_manifest":
        str(
            ROOT
            / "manifests/research_v1/era5_examples.csv"
        ),
    "batch_size": BATCH,
    "stress_examples_per_condition":
        STRESS_N,
    "leads": LEADS,
    "stress_types": STRESS_TYPES,
    "severities": SEVERITIES,
    "stress_conditions":
        len(LEADS)
        * len(STRESS_TYPES)
        * len(SEVERITIES),
    "normalized_metric_definition":
        "z-score using existing training-set channel mean/std",
    "spatial_weighting":
        "cos(latitude)",
    "stress_aggregation":
        "macro mean over 3 leads x 5 corruptions x 4 severities",
    "physical_mixed_rmse_warning":
        "rmse_phys_mixed combines channels with different physical units; "
        "use normalized and per-channel metrics as primary results",
    "models": {
        k: {
            "method": v["method"],
            "checkpoint_dir":
                str(v["checkpoint_dir"]),
        }
        for k, v in MODELS.items()
    },
}

(
    OUT / "protocol.json"
).write_text(
    json.dumps(
        meta,
        indent=2,
    )
)


# ======================================================================
# PRINT FINAL
# ======================================================================

pd.set_option(
    "display.max_columns",
    30,
)

print("\n" + "=" * 120)
print("FINAL COMPARABLE METRICS")
print("=" * 120)

cols = [
    "model",
    "clean_nrmse",
    "clean_acc",
    "macro_stress_nrmse",
    "macro_stress_acc",
    "macro_localization_iou",
    "macro_type_accuracy",
    "macro_repair_l1_norm",
]

print(
    final[cols]
    .sort_values(
        "macro_stress_nrmse"
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\nSaved to:")
print(OUT)
