#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset, DataLoader


ROOT = Path.home() / "ConserveFM"

sys.path.insert(
    0,
    str(ROOT / "external/fcnv2_runtime"),
)

sys.path.insert(
    0,
    str(ROOT / "external/ai-models-fourcastnetv2"),
)

from ai_models_fourcastnetv2.fourcastnetv2 import FourierNeuralOperatorNet
from conservefm.data import ERA5Reader


LEVELS = [
    50, 100, 150, 200, 250, 300, 400,
    500, 600, 700, 850, 925, 1000,
]

FCN_ORDER = (
    [
        "10u", "10v", "100u", "100v",
        "2t", "sp", "msl", "tcwv",
    ]
    + [f"u{x}" for x in LEVELS]
    + [f"v{x}" for x in LEVELS]
    + [f"z{x}" for x in LEVELS]
    + [f"t{x}" for x in LEVELS]
    + [f"r{x}" for x in LEVELS]
)

assert len(FCN_ORDER) == 73

EPS = 0.622
G = 9.80665


def cli():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--era5-zarr",
        default=str(
            ROOT / "data/ERA5_WeatherBench2_1979_2022"
        ),
    )

    p.add_argument(
        "--manifest",
        default=str(
            ROOT / "manifests/research_v1/era5_examples.csv"
        ),
    )

    p.add_argument(
        "--assets",
        default=str(
            ROOT / "models/fourcastnet"
        ),
    )

    p.add_argument(
        "--output",
        default=str(
            ROOT / "forecasts/fourcastnet.zarr"
        ),
    )

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--max-contexts", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def load_requirements(path):
    req = defaultdict(set)

    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lead = int(r["lead_hours"])

            if lead in (6, 24, 72):
                req[int(r["context_end_idx"])].add(lead)

    return {
        k: sorted(v)
        for k, v in req.items()
    }


def create_store(path, ntime, nchan, overwrite):
    path = Path(path)

    if overwrite and path.exists():
        shutil.rmtree(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = zarr.open_group(
        str(path),
        mode="a",
    )

    comp = zarr.Blosc(
        cname="zstd",
        clevel=3,
        shuffle=zarr.Blosc.BITSHUFFLE,
    )

    for lead in (6, 24, 72):
        n = f"lead_{lead}"

        if n not in root:
            root.create_dataset(
                n,
                shape=(ntime, nchan, 240, 121),
                chunks=(1, nchan, 240, 121),
                dtype="f4",
                compressor=comp,
                fill_value=np.nan,
            )

        d = f"_done_{lead}"

        if d not in root:
            root.create_dataset(
                d,
                shape=(ntime,),
                chunks=(4096,),
                dtype="u1",
                fill_value=0,
            )

    root.attrs["adapter_mode"] = (
        "ERA5->FourCastNet-v2-small; "
        "RH/TCWV derived; 100m wind approximated; "
        "precipitation persisted"
    )

    return root


def sat_vapor_pressure(t):
    tc = t - 273.15

    return (
        611.2
        * np.exp(
            17.67 * tc
            / (tc + 243.5)
        )
    )


def q_to_rh(q, t, p):
    q = np.clip(q, 0.0, 0.2)

    e = (
        q * p
        / (
            EPS
            + (1.0 - EPS) * q
        )
    )

    es = np.maximum(
        sat_vapor_pressure(t),
        1.0,
    )

    return np.clip(
        100.0 * e / es,
        0.0,
        100.0,
    )


def rh_to_q(rh, t, p):
    es = sat_vapor_pressure(t)

    e = (
        np.clip(rh, 0.0, 100.0)
        / 100.0
        * es
    )

    den = (
        p
        - (1.0 - EPS) * e
    )

    return np.clip(
        EPS * e
        / np.maximum(den, 1.0),
        0.0,
        0.2,
    )


def periodic_resize(x, h, w):
    #
    # x: [B,C,H,W], longitude is periodic.
    #
    x = torch.cat(
        [
            x,
            x[..., :1],
        ],
        dim=-1,
    )

    x = F.interpolate(
        x,
        size=(h, w + 1),
        mode="bilinear",
        align_corners=True,
    )

    return x[..., :-1]


class InputDataset(Dataset):
    def __init__(
        self,
        reader,
        contexts,
    ):
        self.reader = reader
        self.contexts = contexts

        self.names = list(
            reader.layout.names
        )

        self.index = {
            n: i
            for i, n in enumerate(
                self.names
            )
        }

        lat = np.asarray(
            reader.latitude
        )

        #
        # FCNv2 is north -> south.
        #
        self.flip_lat = (
            lat[0] < lat[-1]
        )

    def __len__(self):
        return len(self.contexts)

    def field(self, state, name):
        #
        # reader: [C,lon,lat]
        # output: [lat,lon]
        #
        return state[
            self.index[name]
        ].T

    def __getitem__(self, n):
        idx = self.contexts[n]

        state = np.asarray(
            self.reader.read_state(idx),
            dtype=np.float32,
        )

        f = lambda name: self.field(
            state,
            name,
        )

        out = np.empty(
            (73, 121, 240),
            dtype=np.float32,
        )

        oi = {
            name: i
            for i, name in enumerate(
                FCN_ORDER
            )
        }

        u10 = f(
            "10m_u_component_of_wind"
        )

        v10 = f(
            "10m_v_component_of_wind"
        )

        out[oi["10u"]] = u10
        out[oi["10v"]] = v10

        #
        # Deterministic neutral-layer approximation.
        #
        wind_factor = (
            100.0 / 10.0
        ) ** 0.14

        out[oi["100u"]] = (
            u10 * wind_factor
        )

        out[oi["100v"]] = (
            v10 * wind_factor
        )

        out[oi["2t"]] = f(
            "2m_temperature"
        )

        out[oi["sp"]] = f(
            "surface_pressure"
        )

        out[oi["msl"]] = f(
            "mean_sea_level_pressure"
        )

        q_stack = np.stack(
            [
                f(
                    f"specific_humidity@{p}hPa"
                )
                for p in LEVELS
            ],
            axis=0,
        )

        pressure = (
            np.asarray(
                LEVELS,
                dtype=np.float32,
            )
            * 100.0
        )

        out[oi["tcwv"]] = (
            np.trapz(
                q_stack,
                pressure,
                axis=0,
            )
            / G
        )

        for p in LEVELS:
            out[
                oi[f"u{p}"]
            ] = f(
                f"u_component_of_wind@{p}hPa"
            )

            out[
                oi[f"v{p}"]
            ] = f(
                f"v_component_of_wind@{p}hPa"
            )

            out[
                oi[f"z{p}"]
            ] = f(
                f"geopotential@{p}hPa"
            )

            t = f(
                f"temperature@{p}hPa"
            )

            q = f(
                f"specific_humidity@{p}hPa"
            )

            out[
                oi[f"t{p}"]
            ] = t

            out[
                oi[f"r{p}"]
            ] = q_to_rh(
                q,
                t,
                p * 100.0,
            )

        if self.flip_lat:
            out = out[
                :,
                ::-1,
                :,
            ].copy()

        tp = state[
            self.index[
                "total_precipitation_6hr"
            ]
        ]

        return {
            "x": torch.from_numpy(out),
            "context_idx": idx,
            "tp": torch.from_numpy(
                tp.copy()
            ),
        }


def map_output(
    fcn,
    tp,
    names,
    flip_lat,
):
    #
    # fcn: [73,121,240]
    #
    if flip_lat:
        fcn = fcn[
            :,
            ::-1,
            :
        ].copy()

    fi = {
        n: i
        for i, n in enumerate(
            FCN_ORDER
        )
    }

    result = np.empty(
        (
            len(names),
            240,
            121,
        ),
        dtype=np.float32,
    )

    ri = {
        n: i
        for i, n in enumerate(
            names
        )
    }

    for p in LEVELS:
        t = fcn[
            fi[f"t{p}"]
        ]

        result[
            ri[f"geopotential@{p}hPa"]
        ] = fcn[
            fi[f"z{p}"]
        ].T

        result[
            ri[f"temperature@{p}hPa"]
        ] = t.T

        result[
            ri[
                f"u_component_of_wind@{p}hPa"
            ]
        ] = fcn[
            fi[f"u{p}"]
        ].T

        result[
            ri[
                f"v_component_of_wind@{p}hPa"
            ]
        ] = fcn[
            fi[f"v{p}"]
        ].T

        result[
            ri[
                f"specific_humidity@{p}hPa"
            ]
        ] = rh_to_q(
            fcn[
                fi[f"r{p}"]
            ],
            t,
            p * 100.0,
        ).T

    result[
        ri["2m_temperature"]
    ] = fcn[
        fi["2t"]
    ].T

    result[
        ri["10m_u_component_of_wind"]
    ] = fcn[
        fi["10u"]
    ].T

    result[
        ri["10m_v_component_of_wind"]
    ] = fcn[
        fi["10v"]
    ].T

    result[
        ri["mean_sea_level_pressure"]
    ] = fcn[
        fi["msl"]
    ].T

    result[
        ri["surface_pressure"]
    ] = fcn[
        fi["sp"]
    ].T

    #
    # FCNv2-small does not provide our
    # 6-hour precipitation field.
    #
    result[
        ri["total_precipitation_6hr"]
    ] = tp

    return result


def load_model(weights, device):
    print(
        "[MODEL] constructing FCNv2-small",
        flush=True,
    )

    model = FourierNeuralOperatorNet()

    ckpt = torch.load(
        weights,
        map_location="cpu",
        weights_only=False,
    )

    raw = ckpt.get(
        "model_state",
        ckpt,
    )

    raw = {
        k: v
        for k, v in raw.items()
        if k not in {
            "module.norm.weight",
            "module.norm.bias",
        }
    }

    sd = {
        (
            k[7:]
            if k.startswith("module.")
            else k
        ): v
        for k, v in raw.items()
    }

    msg = model.load_state_dict(
        sd,
        strict=False,
    )

    print(
        "[MODEL] missing:",
        msg.missing_keys,
        flush=True,
    )

    print(
        "[MODEL] unexpected:",
        msg.unexpected_keys,
        flush=True,
    )

    if len(
        msg.missing_keys
    ) > 4:
        raise RuntimeError(
            "Too many missing FCNv2 weights"
        )

    model.eval()
    model.to(device)

    return model


@torch.inference_mode()
def main():
    args = cli()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    device = torch.device("cuda")

    reader = ERA5Reader(
        args.era5_zarr
    )

    names = list(
        reader.layout.names
    )

    req = load_requirements(
        args.manifest
    )

    contexts = sorted(req)

    if args.max_contexts:
        contexts = contexts[
            :args.max_contexts
        ]

    store = create_store(
        args.output,
        len(reader.times),
        len(names),
        args.overwrite,
    )

    todo = []

    for idx in contexts:
        if not all(
            int(
                store[
                    f"_done_{lead}"
                ][idx]
            ) == 1
            for lead in req[idx]
        ):
            todo.append(idx)

    print(
        f"[DATA] contexts={len(contexts):,} "
        f"remaining={len(todo):,}",
        flush=True,
    )

    if not todo:
        return 0

    ds = InputDataset(
        reader,
        todo,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    assets = Path(
        args.assets
    )

    means_np = np.load(
        assets / "global_means.npy"
    ).astype(np.float32)

    stds_np = np.load(
        assets / "global_stds.npy"
    ).astype(np.float32)

    means_np = means_np.reshape(
        -1
    )[:73].reshape(
        1, 73, 1, 1
    )

    stds_np = stds_np.reshape(
        -1
    )[:73].reshape(
        1, 73, 1, 1
    )

    means = torch.from_numpy(
        means_np
    ).to(device)

    stds = torch.from_numpy(
        stds_np
    ).to(device)

    model = load_model(
        assets / "weights.tar",
        device,
    )

    started = time.time()
    complete = 0

    for batch_no, batch in enumerate(
        loader,
        1,
    ):
        xlow = batch["x"].to(
            device,
            non_blocking=True,
            dtype=torch.float32,
        )

        #
        # Native 0.25 degree grid.
        #
        xhi = periodic_resize(
            xlow,
            721,
            1440,
        )

        x = (
            xhi - means
        ) / stds

        indices = (
            batch["context_idx"]
            .cpu()
            .numpy()
            .astype(int)
        )

        tp = (
            batch["tp"]
            .cpu()
            .numpy()
        )

        max_step = max(
            max(req[int(i)])
            for i in indices
        ) // 6

        for step in range(
            1,
            max_step + 1,
        ):
            x = model(x)

            lead = step * 6

            wanted = [
                j
                for j, idx in enumerate(
                    indices
                )
                if lead in req[int(idx)]
            ]

            if not wanted:
                continue

            #
            # Interpolation is linear, therefore
            # downsample normalized forecast first.
            #
            low_norm = periodic_resize(
                x[wanted],
                121,
                240,
            )

            low_phys = (
                low_norm
                * stds
                + means
            ).float().cpu().numpy()

            for k, bj in enumerate(
                wanted
            ):
                idx = int(
                    indices[bj]
                )

                pred = map_output(
                    low_phys[k],
                    tp[bj],
                    names,
                    ds.flip_lat,
                )

                store[
                    f"lead_{lead}"
                ][idx] = pred

                store[
                    f"_done_{lead}"
                ][idx] = 1

        complete += len(indices)

        if (
            batch_no == 1
            or batch_no
            % args.log_every == 0
        ):
            dt = time.time() - started
            rate = complete / max(dt, 1e-6)

            print(
                f"[FCN] {complete:,}/{len(todo):,} "
                f"{100*complete/len(todo):.2f}% "
                f"rate={rate:.3f} contexts/s "
                f"ETA={(len(todo)-complete)/max(rate,1e-9)/3600:.2f}h "
                f"GPU_peak="
                f"{torch.cuda.max_memory_reserved()/1024**3:.2f}GiB",
                flush=True,
            )

            torch.cuda.reset_peak_memory_stats()

    missing = {}

    for lead in (6, 24, 72):
        ids = [
            idx
            for idx in contexts
            if lead in req[idx]
        ]

        missing[str(lead)] = sum(
            int(
                store[
                    f"_done_{lead}"
                ][i]
            ) != 1
            for i in ids
        )

    summary = {
        "status":
            "completed"
            if not any(missing.values())
            else "incomplete",

        "backbone":
            "fourcastnetv2-small",

        "adapter":
            "era5_1p5_to_native_0p25",

        "contexts":
            len(contexts),

        "missing":
            missing,

        "approximations": [
            "relative humidity derived from ERA5 q,T,p",
            "TCWV pressure-integrated from q",
            "100m winds approximated from 10m winds",
            "total_precipitation_6hr persisted",
        ],
    }

    (
        Path(args.output)
        / "generation_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "[COMPLETE]",
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )

    return (
        0
        if summary["status"]
        == "completed"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
