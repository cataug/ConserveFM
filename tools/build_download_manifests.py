#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path.home() / "ConserveFM"
DATA = ROOT / "data"
OUT = ROOT / "manifests/current"

ERA5 = DATA / "ERA5_WeatherBench2_1979_2022"
GHCN = DATA / "GHCN_Daily_by_year"
ISD = DATA / "ISD_2023_stations"

ERA5_START = np.datetime64("1979-01-01T00:00:00")
ERA5_END_EXCLUSIVE = np.datetime64("2023-01-01T00:00:00")

ERA5_VARIABLES = [
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


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} PiB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = list(rows[0])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_all_files():
    rows = []

    if not DATA.exists():
        return rows

    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue

        stat = path.stat()

        rows.append({
            "dataset_root": (
                path.relative_to(DATA).parts[0]
                if path.relative_to(DATA).parts
                else ""
            ),
            "relative_path": str(path.relative_to(DATA)),
            "size_bytes": stat.st_size,
            "size_human": human_bytes(stat.st_size),
            "mtime": datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(timespec="seconds"),
        })

    return rows


def load_zarray(variable: str):
    path = ERA5 / variable / ".zarray"
    if not path.exists():
        return None

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def numeric_chunk_files(variable: str):
    directory = ERA5 / variable

    result = {}

    if not directory.exists():
        return result

    for path in directory.iterdir():
        if not path.is_file():
            continue

        # Zarr chunk names such as:
        # 3652.0.0.0
        # 3652.0.0
        if not re.fullmatch(
            r"\d+(?:\.\d+)*",
            path.name,
        ):
            continue

        first_index = int(
            path.name.split(".", 1)[0]
        )

        result[path.name] = {
            "path": path,
            "time_chunk": first_index,
            "size": path.stat().st_size,
        }

    return result


def build_era5():
    summary_rows = []
    year_rows = []
    chunk_rows = []

    if not ERA5.exists():
        print("[ERA5] directory not found")
        return summary_rows, year_rows, chunk_rows

    print("[ERA5] reading local Zarr metadata...")

    ds = xr.open_zarr(
        str(ERA5),
        consolidated=False,
        chunks=None,
    )

    times = ds["time"].values

    start_index = int(
        np.searchsorted(
            times,
            ERA5_START,
            side="left",
        )
    )

    end_index = int(
        np.searchsorted(
            times,
            ERA5_END_EXCLUSIVE,
            side="left",
        )
    )

    print(
        f"[ERA5] selected timestamps: "
        f"{end_index - start_index:,}"
    )

    total_expected = 0
    total_present = 0
    total_bytes = 0

    for variable in ERA5_VARIABLES:
        metadata = load_zarray(variable)

        if metadata is None:
            print(
                f"[ERA5] {variable}: metadata missing"
            )
            continue

        chunks = [
            int(x)
            for x in metadata["chunks"]
        ]

        time_chunk_size = chunks[0]

        first_chunk = (
            start_index // time_chunk_size
        )

        last_chunk_exclusive = math.ceil(
            end_index / time_chunk_size
        )

        expected_indices = set(
            range(
                first_chunk,
                last_chunk_exclusive,
            )
        )

        local = numeric_chunk_files(variable)

        present_indices = {
            info["time_chunk"]
            for info in local.values()
            if info["size"] > 0
        }

        expected_count = len(expected_indices)
        present_count = len(
            expected_indices & present_indices
        )

        missing_count = (
            expected_count - present_count
        )

        bytes_present = sum(
            info["size"]
            for info in local.values()
            if info["time_chunk"] in expected_indices
        )

        coverage = (
            100.0 * present_count / expected_count
            if expected_count
            else 0.0
        )

        total_expected += expected_count
        total_present += present_count
        total_bytes += bytes_present

        part_files = list(
            (ERA5 / variable).glob("*.part")
        )

        summary_rows.append({
            "variable": variable,
            "expected_chunks": expected_count,
            "present_chunks": present_count,
            "missing_chunks": missing_count,
            "coverage_percent": round(
                coverage, 4
            ),
            "present_bytes": bytes_present,
            "present_size": human_bytes(
                bytes_present
            ),
            "partial_files": len(part_files),
            "complete": (
                present_count == expected_count
            ),
        })

        print(
            f"[ERA5] {variable:<28} "
            f"{present_count:>5}/{expected_count:<5} "
            f"{coverage:6.2f}%  "
            f"{human_bytes(bytes_present)}"
        )

        #
        # Per-chunk manifest
        #
        for idx in sorted(expected_indices):
            key_candidates = [
                key
                for key, info in local.items()
                if info["time_chunk"] == idx
            ]

            present = bool(key_candidates)

            start_i = idx * time_chunk_size
            end_i = min(
                start_i + time_chunk_size,
                len(times),
            )

            if start_i < len(times):
                chunk_start = str(
                    times[start_i]
                )
            else:
                chunk_start = ""

            if end_i > 0 and end_i <= len(times):
                chunk_end = str(
                    times[end_i - 1]
                )
            else:
                chunk_end = ""

            size = 0
            filename = ""

            if present:
                filename = key_candidates[0]
                size = local[filename]["size"]

            chunk_rows.append({
                "variable": variable,
                "time_chunk_index": idx,
                "chunk_file": filename,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "present": present,
                "size_bytes": size,
                "size_human": (
                    human_bytes(size)
                    if present
                    else ""
                ),
            })

        #
        # Coverage by year
        #
        for year in range(1979, 2023):
            year_start = np.datetime64(
                f"{year}-01-01T00:00:00"
            )
            next_year = np.datetime64(
                f"{year + 1}-01-01T00:00:00"
            )

            yi0 = int(
                np.searchsorted(
                    times,
                    year_start,
                    side="left",
                )
            )

            yi1 = int(
                np.searchsorted(
                    times,
                    next_year,
                    side="left",
                )
            )

            yc0 = yi0 // time_chunk_size
            yc1 = math.ceil(
                yi1 / time_chunk_size
            )

            year_expected = set(
                range(yc0, yc1)
            )

            # restrict to requested ERA5 period
            year_expected &= expected_indices

            year_present = (
                year_expected & present_indices
            )

            n_expected = len(year_expected)
            n_present = len(year_present)

            pct = (
                100.0 * n_present / n_expected
                if n_expected
                else 0.0
            )

            year_rows.append({
                "variable": variable,
                "year": year,
                "expected_chunks": n_expected,
                "present_chunks": n_present,
                "missing_chunks": (
                    n_expected - n_present
                ),
                "coverage_percent": round(
                    pct, 4
                ),
                "complete": (
                    n_expected > 0
                    and n_present == n_expected
                ),
            })

    overall = (
        100.0 * total_present / total_expected
        if total_expected
        else 0
    )

    print()
    print(
        f"[ERA5] OVERALL: "
        f"{total_present:,}/{total_expected:,} "
        f"chunks = {overall:.2f}%"
    )
    print(
        f"[ERA5] stored chunk payload: "
        f"{human_bytes(total_bytes)}"
    )

    return summary_rows, year_rows, chunk_rows


def build_ghcn():
    rows = []

    if not GHCN.exists():
        print("[GHCN] directory not found")
        return rows

    for year in range(1979, 2024):
        path = GHCN / f"{year}.csv.gz"

        exists = (
            path.exists()
            and path.stat().st_size > 0
        )

        size = (
            path.stat().st_size
            if exists
            else 0
        )

        rows.append({
            "year": year,
            "file": path.name,
            "present": exists,
            "size_bytes": size,
            "size_human": (
                human_bytes(size)
                if exists
                else ""
            ),
        })

    complete = sum(
        row["present"]
        for row in rows
    )

    print(
        f"[GHCN] years present: "
        f"{complete}/{len(rows)}"
    )

    return rows


def build_isd():
    rows = []

    if not ISD.exists():
        print("[ISD] directory not found")
        return rows

    files = sorted(
        ISD.glob("*.csv")
    )

    for path in files:
        stat = path.stat()

        station_id = path.stem

        rows.append({
            "station_id": station_id,
            "file": path.name,
            "size_bytes": stat.st_size,
            "size_human": human_bytes(
                stat.st_size
            ),
            "present": stat.st_size > 0,
        })

    part_files = list(
        ISD.glob("*.part")
    )

    print(
        f"[ISD] station files present: "
        f"{len(rows):,}"
    )
    print(
        f"[ISD] partial files: "
        f"{len(part_files):,}"
    )

    return rows


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("ConserveFM downloaded-data manifest")
    print("Data root:", DATA)
    print("Output:", OUT)
    print("=" * 72)

    all_files = build_all_files()

    era5_summary, era5_years, era5_chunks = (
        build_era5()
    )

    ghcn_rows = build_ghcn()
    isd_rows = build_isd()

    write_csv(
        OUT / "all_files.csv",
        all_files,
    )

    write_csv(
        OUT / "era5_summary.csv",
        era5_summary,
    )

    write_csv(
        OUT / "era5_by_year.csv",
        era5_years,
    )

    write_csv(
        OUT / "era5_chunks.csv",
        era5_chunks,
    )

    write_csv(
        OUT / "ghcn_years.csv",
        ghcn_rows,
    )

    write_csv(
        OUT / "isd_stations.csv",
        isd_rows,
    )

    era5_expected = sum(
        x["expected_chunks"]
        for x in era5_summary
    )

    era5_present = sum(
        x["present_chunks"]
        for x in era5_summary
    )

    summary = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "project_root": str(ROOT),
        "data_root": str(DATA),

        "datasets": {
            "ERA5_WeatherBench2_1979_2022": {
                "path": str(ERA5),
                "disk_size_bytes": dir_size(ERA5),
                "disk_size_human": human_bytes(
                    dir_size(ERA5)
                ),
                "expected_chunks": era5_expected,
                "present_chunks": era5_present,
                "missing_chunks": (
                    era5_expected - era5_present
                ),
                "coverage_percent": (
                    round(
                        100.0
                        * era5_present
                        / era5_expected,
                        4,
                    )
                    if era5_expected
                    else 0
                ),
                "variables": len(
                    era5_summary
                ),
            },

            "GHCN_Daily_by_year": {
                "path": str(GHCN),
                "disk_size_bytes": dir_size(GHCN),
                "disk_size_human": human_bytes(
                    dir_size(GHCN)
                ),
                "years_expected": len(
                    ghcn_rows
                ),
                "years_present": sum(
                    int(x["present"])
                    for x in ghcn_rows
                ),
            },

            "ISD_2023_stations": {
                "path": str(ISD),
                "disk_size_bytes": dir_size(ISD),
                "disk_size_human": human_bytes(
                    dir_size(ISD)
                ),
                "station_files_present": len(
                    isd_rows
                ),
            },
        },

        "total_data_size_bytes": dir_size(DATA),
        "total_data_size_human": human_bytes(
            dir_size(DATA)
        ),
    }

    (
        OUT
        / "inventory_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("MANIFESTS CREATED")

    for path in sorted(OUT.iterdir()):
        print(
            f"  {path.name:<25} "
            f"{human_bytes(path.stat().st_size)}"
        )

    print()
    print(
        "Total data on disk:",
        summary["total_data_size_human"],
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
