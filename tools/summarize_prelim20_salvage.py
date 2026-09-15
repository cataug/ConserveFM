from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path.home() / "ConserveFM"

META = pd.read_csv(
    ROOT
    / "reports/prelim20_salvage/jobs.csv"
)

EVAL = ROOT / "runs/prelim20_eval"

OUT = ROOT / "reports/prelim20_salvage"

meta = {
    r.run_id: r
    for _, r in META.iterrows()
}


def mean_metric(d, key):
    vals = []

    for v in d.values():

        if (
            isinstance(v, dict)
            and key in v
        ):
            try:
                vals.append(
                    float(v[key])
                )
            except Exception:
                pass

    return (
        float(np.mean(vals))
        if vals
        else np.nan
    )


rows = []

for d in sorted(EVAL.iterdir()):

    if not d.is_dir():
        continue

    p = d / "metrics_stress.json"

    if not p.exists():
        continue

    s = json.loads(
        p.read_text()
    )

    run_id = d.name

    if run_id.startswith("det_"):

        x = run_id.split("_")

        # det_prithvi_wxc_raw_fm needs special handling.
        if run_id.startswith(
            "det_prithvi_wxc_"
        ):
            backbone = "prithvi_wxc"
            method = run_id.replace(
                "det_prithvi_wxc_",
                "",
            )
        else:
            backbone = x[1]
            method = "_".join(x[2:])

        seed = 42

        source_kind = "deterministic"

        prior_step = 0
        prior_val = np.nan

    else:

        if run_id not in meta:
            continue

        r = meta[run_id]

        backbone = r.backbone
        method = r.method
        seed = int(r.seed)

        source_kind = r.source_kind

        prior_step = r.prior_global_step
        prior_val = r.prior_best_val

    rows.append({
        "run_id":
            run_id,

        "backbone":
            backbone,

        "method":
            method,

        "seed":
            seed,

        "source_kind":
            source_kind,

        "prior_global_step":
            prior_step,

        "prior_best_val":
            prior_val,

        # IMPORTANT:
        # preliminary mixed-unit diagnostic only.
        "macro_stress_rmse_phys_mixed":
            mean_metric(
                s,
                "rmse_phys",
            ),

        "macro_stress_mae_phys_mixed":
            mean_metric(
                s,
                "mae_phys",
            ),

        "localization_iou":
            mean_metric(
                s,
                "localization_iou",
            ),

        "type_accuracy":
            mean_metric(
                s,
                "type_accuracy",
            ),

        "repair_l1_norm":
            mean_metric(
                s,
                "repair_l1_norm",
            ),

        "physics_moisture":
            mean_metric(
                s,
                "physics_moisture",
            ),

        "physics_hydrostatic":
            mean_metric(
                s,
                "physics_hydrostatic",
            ),

        "physics_advection":
            mean_metric(
                s,
                "physics_advection",
            ),

        "physics_spectral":
            mean_metric(
                s,
                "physics_spectral",
            ),
    })


all_runs = pd.DataFrame(rows)

all_runs.to_csv(
    OUT / "prelim20_all_runs.csv",
    index=False,
)


# ============================================================
# GROUP ACROSS SEEDS
# ============================================================

metrics = [
    "macro_stress_rmse_phys_mixed",
    "macro_stress_mae_phys_mixed",
    "localization_iou",
    "type_accuracy",
    "repair_l1_norm",
    "physics_moisture",
    "physics_hydrostatic",
    "physics_advection",
    "physics_spectral",
]

agg = (
    all_runs
    .groupby(
        ["backbone", "method"]
    )[metrics]
    .agg(
        ["mean", "std", "count"]
    )
)

agg.to_csv(
    OUT / "prelim20_method_summary.csv"
)


# Flatten for easy sorting.
flat = (
    all_runs
    .groupby(
        ["backbone", "method"]
    )
    .agg(
        n=("run_id", "count"),

        stress_rmse=(
            "macro_stress_rmse_phys_mixed",
            "mean",
        ),

        stress_rmse_std=(
            "macro_stress_rmse_phys_mixed",
            "std",
        ),

        stress_mae=(
            "macro_stress_mae_phys_mixed",
            "mean",
        ),

        loc_iou=(
            "localization_iou",
            "mean",
        ),

        type_acc=(
            "type_accuracy",
            "mean",
        ),

        prior_step_mean=(
            "prior_global_step",
            "mean",
        ),
    )
    .reset_index()
)

flat.to_csv(
    OUT / "prelim20_ranking.csv",
    index=False,
)


print(
    "\n"
    + "=" * 115
)

print(
    "PRELIMINARY SCREEN — CLIMAX"
)

print(
    "same 16-example × 60-condition stress subset"
)

print(
    "=" * 115
)

q = (
    flat[
        flat.backbone == "climax"
    ]
    .sort_values(
        "stress_rmse"
    )
)

print(
    q[
        [
            "method",
            "n",
            "stress_rmse",
            "stress_rmse_std",
            "stress_mae",
            "loc_iou",
            "type_acc",
            "prior_step_mean",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print(
    "\n"
    + "=" * 115
)

print(
    "CROSS-BACKBONE PRELIMINARY"
)

print(
    "=" * 115
)

print(
    flat.sort_values(
        [
            "backbone",
            "stress_rmse",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print(
    "\nSaved:"
)

print(
    OUT
    / "prelim20_all_runs.csv"
)

print(
    OUT
    / "prelim20_ranking.csv"
)
