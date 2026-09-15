#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path.home() / "ConserveFM"

DATASET = (
    ROOT
    / "data/ERA5_WeatherBench2_1979_2022"
)

MANIFEST = (
    ROOT
    / "manifests/era5_remaining_aria2.txt"
)

REMOTE_ROOT = (
    "https://storage.googleapis.com/"
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_"
    "equiangular_with_poles_conservative.zarr"
)

START = np.datetime64("1979-01-01T00:00:00")
END_EXCLUSIVE = np.datetime64("2023-01-01T00:00:00")

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


def read_zarray(variable: str) -> dict:
    path = DATASET / variable / ".zarray"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing local Zarr metadata: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def is_complete(path: Path) -> bool:
    control = Path(str(path) + ".aria2")

    return (
        path.exists()
        and path.stat().st_size > 0
        and not control.exists()
    )


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"ERA5 directory not found: {DATASET}"
        )

    print("=" * 72)
    print("Building aria2 manifest for ERA5")
    print("Local dataset:", DATASET)
    print("Period: 1979-01-01 through 2022-12-31")
    print("=" * 72)

    dataset = xr.open_zarr(
        str(DATASET),
        consolidated=False,
        chunks=None,
    )

    times = dataset["time"].values

    start_index = int(
        np.searchsorted(
            times,
            START,
            side="left",
        )
    )

    end_index = int(
        np.searchsorted(
            times,
            END_EXCLUSIVE,
            side="left",
        )
    )

    print("Start index:", start_index)
    print("End index:", end_index)
    print(
        "Selected timestamps:",
        end_index - start_index,
    )

    MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_objects = 0
    completed_objects = 0
    remaining_objects = 0

    with MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as output:
        for variable in VARIABLES:
            metadata = read_zarray(variable)

            shape = [
                int(value)
                for value in metadata["shape"]
            ]

            chunks = [
                int(value)
                for value in metadata["chunks"]
            ]

            separator = metadata.get(
                "dimension_separator",
                ".",
            )

            first_time_chunk = (
                start_index // chunks[0]
            )

            last_time_chunk_exclusive = math.ceil(
                end_index / chunks[0]
            )

            other_chunk_counts = [
                math.ceil(
                    dimension_size / chunk_size
                )
                for dimension_size, chunk_size
                in zip(shape[1:], chunks[1:])
            ]

            variable_total = 0
            variable_complete = 0
            variable_remaining = 0

            for time_chunk in range(
                first_time_chunk,
                last_time_chunk_exclusive,
            ):
                ranges = [
                    range(count)
                    for count in other_chunk_counts
                ]

                for other_indices in itertools.product(
                    *ranges
                ):
                    indices = (
                        time_chunk,
                        *other_indices,
                    )

                    key = separator.join(
                        str(index)
                        for index in indices
                    )

                    local_path = (
                        DATASET
                        / variable
                        / key
                    )

                    variable_total += 1
                    total_objects += 1

                    if is_complete(local_path):
                        variable_complete += 1
                        completed_objects += 1
                        continue

                    local_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    url = (
                        f"{REMOTE_ROOT}/"
                        f"{variable}/{key}"
                    )

                    output.write(url + "\n")
                    output.write(
                        f"  dir={local_path.parent}\n"
                    )
                    output.write(
                        f"  out={local_path.name}\n"
                    )
                    output.write(
                        "  continue=true\n"
                    )

                    variable_remaining += 1
                    remaining_objects += 1

            print()
            print(variable)
            print(
                f"  total:     {variable_total:,}"
            )
            print(
                f"  complete:  {variable_complete:,}"
            )
            print(
                f"  remaining: {variable_remaining:,}"
            )

    print()
    print("=" * 72)
    print(
        f"Total expected objects: {total_objects:,}"
    )
    print(
        f"Already complete:       {completed_objects:,}"
    )
    print(
        f"Remaining:              {remaining_objects:,}"
    )
    print("Manifest:", MANIFEST)
    print("=" * 72)


if __name__ == "__main__":
    main()
