#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path.home() / "ConserveFM"
ELEMENTS = ["TAVG", "TMAX", "TMIN", "PRCP", "AWND", "ASLP", "ASTP"]
ELEMENT_TO_ID = {e: i for i, e in enumerate(ELEMENTS)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "manifests/research_v1/ghcn_val_test_observations.csv.gz",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests/research_v1/ghcn_grid_eval_2019_2022.npz",
    )
    return p.parse_args()


def ordinal(yyyymmdd: str) -> int:
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])).toordinal()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. First build it with BUILD_GHCN_OBS=1 tools/build_research_manifests.py"
        )

    # Streaming sum/count aggregate to ERA5 grid cell per day and variable.
    # Four years are manageable in this representation and it avoids re-reading raw GHCN for every model.
    agg = defaultdict(lambda: [0.0, 0])
    n = 0
    with gzip.open(args.input, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            elem = row["element"]
            if elem not in ELEMENT_TO_ID:
                continue
            key = (
                ordinal(row["date"]),
                int(row["era5_lat_idx"]),
                int(row["era5_lon_idx"]),
                ELEMENT_TO_ID[elem],
            )
            a = agg[key]
            a[0] += float(row["value"])
            a[1] += 1
            n += 1
            if n % 1_000_000 == 0:
                print(f"[GHCN CACHE] read={n:,} aggregated_cells={len(agg):,}", flush=True)

    keys = sorted(agg)
    day = np.empty(len(keys), dtype=np.int32)
    lat = np.empty(len(keys), dtype=np.int16)
    lon = np.empty(len(keys), dtype=np.int16)
    elem = np.empty(len(keys), dtype=np.int8)
    value = np.empty(len(keys), dtype=np.float32)
    count = np.empty(len(keys), dtype=np.int16)

    for i, key in enumerate(keys):
        d, la, lo, e = key
        s, c = agg[key]
        day[i] = d
        lat[i] = la
        lon[i] = lo
        elem[i] = e
        value[i] = s / c
        count[i] = min(c, np.iinfo(np.int16).max)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        day=day,
        lat_idx=lat,
        lon_idx=lon,
        element=elem,
        value=value,
        station_count=count,
        element_names=np.asarray(ELEMENTS, dtype=object),
    )
    print(f"[GHCN CACHE] source observations={n:,}", flush=True)
    print(f"[GHCN CACHE] aggregated records={len(keys):,}", flush=True)
    print(f"[GHCN CACHE] saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
