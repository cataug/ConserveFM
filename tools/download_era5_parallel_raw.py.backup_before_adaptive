#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
import math
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr
import zarr


SOURCE = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_"
    "equiangular_with_poles_conservative.zarr"
)

DESTINATION = (
    Path.home()
    / "ConserveFM/data/ERA5_WeatherBench2_1979_2022"
)

START_TIME = np.datetime64("1979-01-01T00:00:00")
END_EXCLUSIVE = np.datetime64("2023-01-01T00:00:00")

WORKERS = int(os.environ.get("ERA5_WORKERS", "32"))
MAX_RETRIES = 30

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

COORDINATES = [
    "time",
    "level",
    "longitude",
    "latitude",
]

PRINT_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    with PRINT_LOCK:
        print(f"{now()} [ERA5] {message}", flush=True)


def human_bytes(value: int | float) -> str:
    value = float(value)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024:
            return f"{value:,.2f} {unit}"
        value /= 1024

    return f"{value:,.2f} PiB"


def get_thread_filesystem() -> gcsfs.GCSFileSystem:
    if not hasattr(THREAD_LOCAL, "filesystem"):
        THREAD_LOCAL.filesystem = gcsfs.GCSFileSystem(
            token="anon",
            retries=10,
            cache_timeout=0,
        )

    return THREAD_LOCAL.filesystem


def local_path(remote_path: str) -> Path:
    relative = remote_path.removeprefix(SOURCE).lstrip("/")
    return DESTINATION / relative


def download_one(remote_path: str) -> tuple[int, bool]:
    destination = local_path(remote_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Окончательный файл появляется только после полностью успешной загрузки.
    if destination.exists():
        return destination.stat().st_size, True

    partial = Path(str(destination) + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if partial.exists():
                partial.unlink()

            filesystem = get_thread_filesystem()

            filesystem.get_file(
                remote_path,
                str(partial),
            )

            if not partial.exists():
                raise RuntimeError("temporary file was not created")

            size = partial.stat().st_size

            if size <= 0:
                raise RuntimeError("downloaded object is empty")

            os.replace(partial, destination)
            return size, False

        except Exception as error:
            delay = min(60, 2 + attempt * 2)

            log(
                f"retry {attempt}/{MAX_RETRIES}: "
                f"{remote_path.rsplit('/', 2)[-2:]} | "
                f"{type(error).__name__}: {error}"
            )

            time.sleep(delay)

    raise RuntimeError(
        f"failed after {MAX_RETRIES} attempts: {remote_path}"
    )


def read_zarray(
    filesystem: gcsfs.GCSFileSystem,
    variable: str,
) -> dict:
    path = f"{SOURCE}/{variable}/.zarray"
    return json.loads(filesystem.cat_file(path))


def build_data_tasks(
    filesystem: gcsfs.GCSFileSystem,
    start_index: int,
    end_index: int,
) -> list[str]:
    tasks: list[str] = []

    for variable in VARIABLES:
        metadata = read_zarray(filesystem, variable)

        shape = metadata["shape"]
        chunks = metadata["chunks"]
        separator = metadata.get("dimension_separator", ".")

        if len(shape) != len(chunks):
            raise RuntimeError(
                f"invalid Zarr metadata for {variable}"
            )

        time_chunk = int(chunks[0])

        first_time_chunk = start_index // time_chunk
        last_time_chunk_exclusive = math.ceil(
            end_index / time_chunk
        )

        non_time_chunk_counts = [
            math.ceil(int(size) / int(chunk))
            for size, chunk in zip(
                shape[1:],
                chunks[1:],
            )
        ]

        non_time_ranges = [
            range(count)
            for count in non_time_chunk_counts
        ]

        variable_count = 0

        for time_chunk_index in range(
            first_time_chunk,
            last_time_chunk_exclusive,
        ):
            for remaining_indices in itertools.product(
                *non_time_ranges
            ):
                indices = (
                    time_chunk_index,
                    *remaining_indices,
                )

                key = separator.join(
                    str(index)
                    for index in indices
                )

                tasks.append(
                    f"{SOURCE}/{variable}/{key}"
                )

                variable_count += 1

        log(
            f"{variable}: source chunks "
            f"{first_time_chunk}.."
            f"{last_time_chunk_exclusive - 1} | "
            f"objects={variable_count:,}"
        )

    return tasks


def copy_metadata(
    filesystem: gcsfs.GCSFileSystem,
) -> None:
    log("Copying Zarr metadata and coordinates")

    metadata_paths = [
        f"{SOURCE}/.zgroup",
        f"{SOURCE}/.zattrs",
    ]

    for coordinate in COORDINATES:
        paths = filesystem.find(
            f"{SOURCE}/{coordinate}",
            withdirs=False,
            detail=False,
        )

        metadata_paths.extend(paths)

    for variable in VARIABLES:
        metadata_paths.extend(
            [
                f"{SOURCE}/{variable}/.zarray",
                f"{SOURCE}/{variable}/.zattrs",
            ]
        )

    for index, path in enumerate(
        metadata_paths,
        start=1,
    ):
        size, skipped = download_one(path)

        state = "exists" if skipped else "downloaded"

        log(
            f"metadata {index}/{len(metadata_paths)}: "
            f"{path.rsplit('/', 1)[-1]} | "
            f"{state} | {human_bytes(size)}"
        )


def run_parallel_download(tasks: list[str]) -> None:
    total = len(tasks)

    completed = 0
    skipped = 0
    bytes_this_run = 0
    started = time.time()
    last_report = 0.0

    log(f"Starting raw-object copy with {WORKERS} workers")
    log(f"Total ERA5 chunk objects: {total:,}")
    log(f"Destination: {DESTINATION}")

    task_iterator = iter(tasks)
    pending = {}

    with ThreadPoolExecutor(
        max_workers=WORKERS,
        thread_name_prefix="era5",
    ) as executor:
        # Ограничиваем число одновременно созданных Future.
        for _ in range(WORKERS * 4):
            try:
                remote_path = next(task_iterator)
            except StopIteration:
                break

            future = executor.submit(
                download_one,
                remote_path,
            )
            pending[future] = remote_path

        while pending:
            finished, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in finished:
                remote_path = pending.pop(future)

                size, was_skipped = future.result()

                completed += 1

                if was_skipped:
                    skipped += 1
                else:
                    bytes_this_run += size

                try:
                    next_remote = next(task_iterator)
                except StopIteration:
                    next_remote = None

                if next_remote is not None:
                    next_future = executor.submit(
                        download_one,
                        next_remote,
                    )
                    pending[next_future] = next_remote

            current = time.time()

            if (
                current - last_report >= 30
                or completed == total
            ):
                elapsed = max(
                    current - started,
                    0.001,
                )

                speed = bytes_this_run / elapsed
                percent = completed / total * 100

                remaining_files = total - completed

                if completed - skipped > 0:
                    average_file_size = (
                        bytes_this_run
                        / (completed - skipped)
                    )
                    estimated_remaining_bytes = (
                        average_file_size
                        * remaining_files
                    )
                    eta_seconds = (
                        estimated_remaining_bytes / speed
                        if speed > 0
                        else 0
                    )
                else:
                    eta_seconds = 0

                log(
                    f"progress={completed:,}/{total:,} "
                    f"({percent:.2f}%) | "
                    f"new={human_bytes(bytes_this_run)} | "
                    f"skipped={skipped:,} | "
                    f"average={human_bytes(speed)}/s | "
                    f"ETA≈{eta_seconds / 3600:.1f} h"
                )

                last_report = current


def validate_local_store() -> None:
    log("Creating consolidated local metadata")

    group = zarr.open_group(
        str(DESTINATION),
        mode="a",
    )

    group.attrs[
        "conservefm_local_available_time_start"
    ] = "1979-01-01T00:00:00"

    group.attrs[
        "conservefm_local_available_time_end"
    ] = "2022-12-31T18:00:00"

    group.attrs[
        "conservefm_sparse_outside_available_range"
    ] = True

    zarr.consolidate_metadata(
        str(DESTINATION)
    )

    marker = DESTINATION / "AVAILABLE_TIME_RANGE.txt"

    marker.write_text(
        "Locally downloaded ERA5 data chunks:\n"
        "1979-01-01T00:00:00 through "
        "2022-12-31T18:00:00\n\n"
        "The original Zarr metadata retains the complete "
        "1959-2023 coordinate range. Data chunks outside "
        "1979-2022 were intentionally not downloaded.\n",
        encoding="utf-8",
    )

    local = xr.open_zarr(
        str(DESTINATION),
        consolidated=True,
        chunks=None,
    )

    selected = local[VARIABLES].sel(
        time=slice(
            "1979-01-01T00:00:00",
            "2022-12-31T18:00:00",
        )
    )

    log(f"Validation dimensions: {dict(selected.sizes)}")

    # Читаем по одной локальной точке с начала и конца диапазона.
    first = float(
        selected["2m_temperature"]
        .isel(
            time=0,
            longitude=0,
            latitude=0,
        )
        .values
    )

    last = float(
        selected["2m_temperature"]
        .isel(
            time=-1,
            longitude=0,
            latitude=0,
        )
        .values
    )

    log(
        f"Local data check passed: "
        f"first={first:.4f}, last={last:.4f}"
    )

    (DESTINATION / "_SUCCESS").touch()


def main() -> None:
    DESTINATION.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 72)
    log("Parallel raw WeatherBench2 ERA5 downloader")
    log("Period: 1979–2022")
    log(f"Workers: {WORKERS}")
    log("No decompression or recompression")
    log("=" * 72)

    filesystem = gcsfs.GCSFileSystem(
        token="anon",
        retries=10,
        cache_timeout=0,
    )

    log("Opening remote metadata")

    dataset = xr.open_zarr(
        filesystem.get_mapper(SOURCE),
        consolidated=True,
        chunks=None,
    )

    time_values = dataset["time"].values

    start_index = int(
        np.searchsorted(
            time_values,
            START_TIME,
            side="left",
        )
    )

    end_index = int(
        np.searchsorted(
            time_values,
            END_EXCLUSIVE,
            side="left",
        )
    )

    log(
        f"Selected time indices: "
        f"{start_index:,}..{end_index - 1:,}"
    )

    log(
        f"Selected timestamps: "
        f"{end_index - start_index:,}"
    )

    copy_metadata(filesystem)

    tasks = build_data_tasks(
        filesystem,
        start_index,
        end_index,
    )

    run_parallel_download(tasks)
    validate_local_store()

    log("=" * 72)
    log("ERA5 DOWNLOAD COMPLETE")
    log(f"Output: {DESTINATION}")
    log("=" * 72)


if __name__ == "__main__":
    main()
