#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr
import zarr


DEFAULT_ROOT = (
    Path.home()
    / "ConserveFM/data/ERA5_WeatherBench2_1979_2022"
)

REMOTE = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_"
    "equiangular_with_poles_conservative.zarr"
)

VARIABLES = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation_6hr",
]


def open_xr_local(root):
    try:
        return xr.open_zarr(
            str(root),
            consolidated=True,
            chunks=None,
        )
    except Exception:
        return xr.open_zarr(
            str(root),
            consolidated=False,
            chunks=None,
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
    )

    ap.add_argument(
        "--repair",
        action="store_true",
    )

    ap.add_argument(
        "--progress-every",
        type=int,
        default=250,
    )

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    print("=" * 76)
    print("ERA5 ZARR INTEGRITY CHECK")
    print("=" * 76)
    print("root   :", root)
    print("repair :", args.repair)

    local_xr = open_xr_local(root)
    zg = zarr.open_group(
        str(root),
        mode="r+" if args.repair else "r",
    )

    local_times = np.asarray(
        local_xr["time"].values
    )

    remote_xr = None

    def get_remote():
        nonlocal remote_xr

        if remote_xr is None:
            print(
                "[REMOTE] opening public WeatherBench2 GCS",
                flush=True,
            )

            fs = gcsfs.GCSFileSystem(
                token="anon"
            )

            mapper = fs.get_mapper(
                REMOTE
            )

            try:
                remote_xr = xr.open_zarr(
                    mapper,
                    consolidated=True,
                    chunks=None,
                )
            except Exception:
                remote_xr = xr.open_zarr(
                    mapper,
                    consolidated=False,
                    chunks=None,
                )

        return remote_xr

    total_checks = 0

    for var in VARIABLES:
        za = zg[var]

        tc = int(
            za.chunks[0]
        )

        total_checks += math.ceil(
            za.shape[0] / tc
        )

        print(
            f"[INFO] {var:<30} "
            f"shape={za.shape} chunks={za.chunks}",
            flush=True,
        )

    print(
        f"[INFO] total time chunks = {total_checks:,}",
        flush=True,
    )

    checked = 0
    bad = 0
    repaired = 0
    failed = 0

    started = time.time()

    for var in VARIABLES:
        za = zg[var]

        time_chunk = int(
            za.chunks[0]
        )

        n_time_chunks = math.ceil(
            za.shape[0]
            / time_chunk
        )

        dims = tuple(
            local_xr[var].dims
        )

        print()
        print(
            f"[VARIABLE] {var} "
            f"chunks={n_time_chunks:,}",
            flush=True,
        )

        for ci in range(
            n_time_chunks
        ):
            start = (
                ci * time_chunk
            )

            end = min(
                start + time_chunk,
                za.shape[0],
            )

            checked += 1

            try:
                #
                # Force actual decompression.
                #
                block = np.asarray(
                    za[start:end, ...]
                )

                del block

            except Exception as exc:
                bad += 1

                t0 = local_times[
                    start
                ]

                t1 = local_times[
                    end - 1
                ]

                print()
                print(
                    f"[BAD] var={var} "
                    f"chunk={ci} "
                    f"indices={start}:{end} "
                    f"time={t0}..{t1}",
                    flush=True,
                )

                print(
                    f"[BAD] {type(exc).__name__}: {exc}",
                    flush=True,
                )

                if not args.repair:
                    continue

                try:
                    remote = get_remote()

                    times = local_times[
                        start:end
                    ]

                    source = remote[
                        var
                    ].sel(
                        time=xr.DataArray(
                            times,
                            dims="time",
                        )
                    )

                    #
                    # Guarantee identical dimension order.
                    #
                    source = source.transpose(
                        *dims
                    )

                    data = np.asarray(
                        source.values,
                        dtype=za.dtype,
                    )

                    expected = (
                        end - start,
                        *za.shape[1:],
                    )

                    if data.shape != expected:
                        raise RuntimeError(
                            f"remote shape {data.shape} "
                            f"!= local expected {expected}"
                        )

                    #
                    # Full time-chunk replacement.
                    #
                    za[start:end, ...] = data

                    #
                    # Immediately verify fresh local bytes.
                    #
                    test = np.asarray(
                        za[start:end, ...]
                    )

                    del data
                    del test

                    repaired += 1

                    print(
                        f"[REPAIRED] {var} "
                        f"chunk={ci}",
                        flush=True,
                    )

                except Exception as repair_exc:
                    failed += 1

                    print(
                        f"[REPAIR FAILED] "
                        f"{type(repair_exc).__name__}: "
                        f"{repair_exc}",
                        flush=True,
                    )

            if (
                checked % args.progress_every
                == 0
            ):
                elapsed = (
                    time.time()
                    - started
                )

                speed = (
                    checked / elapsed
                    if elapsed > 0
                    else 0
                )

                eta = (
                    (total_checks - checked)
                    / speed
                    if speed > 0
                    else 0
                )

                print(
                    f"[PROGRESS] "
                    f"{checked:,}/{total_checks:,} "
                    f"{100*checked/total_checks:.2f}% "
                    f"bad={bad} "
                    f"repaired={repaired} "
                    f"failed={failed} "
                    f"speed={speed:.1f} chunks/s "
                    f"ETA={eta/60:.1f} min",
                    flush=True,
                )

    elapsed = (
        time.time()
        - started
    )

    print()
    print("=" * 76)
    print("INTEGRITY RESULT")
    print("=" * 76)
    print(
        f"checked  : {checked:,}"
    )
    print(
        f"bad      : {bad:,}"
    )
    print(
        f"repaired : {repaired:,}"
    )
    print(
        f"failed   : {failed:,}"
    )
    print(
        f"elapsed  : {elapsed/60:.1f} min"
    )

    if failed:
        print("STATUS   : FAILED")
        return 2

    print("STATUS   : OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
