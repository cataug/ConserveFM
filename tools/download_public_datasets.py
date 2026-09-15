#!/usr/bin/env python3

import multiprocessing as mp
import os
import shutil
import signal
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path.home() / "ConserveFM"
DATA = ROOT / "data"

ERA5_DIR = DATA / "ERA5_1p5deg_1979_2022"
GHCN_DIR = DATA / "GHCN_Daily"
WEATHERREAL_DIR = DATA / "WeatherReal_2023"

ERA5_SOURCE = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_"
    "equiangular_with_poles_conservative.zarr"
)

GHCN_URL = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
    "ghcnd_all.tar.gz"
)

WEATHERREAL_URL = (
    "https://media.githubusercontent.com/media/"
    "microsoft/WeatherReal/main/Data/"
    "WeatherReal-ISD-2023.nc"
)

ERA5_VARIABLES = [
    # 5 atmospheric variables × all 13 pressure levels
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",

    # 6 surface variables
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation_6hr",
]


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(name, message):
    for line in str(message).splitlines():
        print(
            f"{timestamp()} [{name:<11}] {line}",
            flush=True,
        )


def human_bytes(value):
    value = float(value)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PiB"


def directory_size(path):
    path = Path(path)

    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    total = 0

    for root, _, files in os.walk(path):
        for filename in files:
            candidate = Path(root) / filename

            try:
                total += candidate.stat().st_size
            except OSError:
                pass

    return total


def free_space():
    DATA.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(DATA).free


def download_http(name, url, destination, minimum_size):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    partial = Path(str(destination) + ".part")

    if destination.exists() and destination.stat().st_size >= minimum_size:
        log(
            name,
            f"Already downloaded: {destination} "
            f"({human_bytes(destination.stat().st_size)})",
        )
        return

    existing = partial.stat().st_size if partial.exists() else 0

    headers = {
        "User-Agent": "Mozilla/5.0 ConserveFM downloader",
        "Accept": "*/*",
    }

    if existing:
        headers["Range"] = f"bytes={existing}-"
        log(
            name,
            f"Resuming from {human_bytes(existing)}",
        )
    else:
        log(name, "Starting new download")

    log(name, f"URL: {url}")
    log(name, f"Destination: {destination}")

    request = urllib.request.Request(
        url,
        headers=headers,
    )

    response = urllib.request.urlopen(
        request,
        timeout=180,
    )

    status = getattr(response, "status", 200)

    if existing and status == 206:
        mode = "ab"
        downloaded = existing
    else:
        mode = "wb"
        downloaded = 0

        if existing:
            log(
                name,
                "Server did not accept resume; restarting file",
            )

    content_length = response.headers.get("Content-Length")
    content_range = response.headers.get("Content-Range")

    total = None

    if content_range and "/" in content_range:
        final_part = content_range.rsplit("/", 1)[-1]

        if final_part.isdigit():
            total = int(final_part)

    elif content_length and content_length.isdigit():
        total = downloaded + int(content_length)

    if total:
        log(name, f"Remote size: {human_bytes(total)}")
    else:
        log(name, "Remote server did not report total size")

    started = time.time()
    last_report = 0

    with partial.open(mode) as output:
        while True:
            block = response.read(8 * 1024 * 1024)

            if not block:
                break

            output.write(block)
            downloaded += len(block)

            current = time.time()

            if current - last_report >= 15:
                elapsed = max(current - started, 0.001)
                session_bytes = downloaded - existing
                speed = session_bytes / elapsed

                if total:
                    percent = downloaded / total * 100

                    log(
                        name,
                        f"{human_bytes(downloaded)} / "
                        f"{human_bytes(total)} "
                        f"({percent:.1f}%) | "
                        f"{human_bytes(speed)}/s",
                    )
                else:
                    log(
                        name,
                        f"{human_bytes(downloaded)} | "
                        f"{human_bytes(speed)}/s",
                    )

                last_report = current

    final_size = partial.stat().st_size

    if final_size < minimum_size:
        raise RuntimeError(
            f"Downloaded file is unexpectedly small: "
            f"{human_bytes(final_size)}"
        )

    partial.rename(destination)

    log(
        name,
        f"COMPLETE: {destination} "
        f"({human_bytes(final_size)})",
    )


def download_ghcn():
    download_http(
        name="GHCN",
        url=GHCN_URL,
        destination=GHCN_DIR / "ghcnd_all.tar.gz",
        minimum_size=1_000_000_000,
    )


def download_weatherreal():
    download_http(
        name="WEATHERREAL",
        url=WEATHERREAL_URL,
        destination=(
            WEATHERREAL_DIR /
            "WeatherReal-ISD-2023.nc"
        ),
        minimum_size=1_000_000,
    )


def raw_size(dataset):
    total = 0

    for name in dataset.data_vars:
        array = dataset[name]
        elements = 1

        for dimension in array.shape:
            elements *= int(dimension)

        total += elements * array.dtype.itemsize

    return total


def download_era5():
    import dask
    import gcsfs
    import xarray as xr
    from numcodecs import Blosc

    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    log("ERA5", "Opening public WeatherBench2 archive")
    log("ERA5", f"Source: gs://{ERA5_SOURCE}")
    log("ERA5", "Period: 1979-01-01 to 2022-12-31")
    log("ERA5", "Resolution: 240 × 121, every 6 hours")
    log("ERA5", "Pressure levels: all 13")
    log("ERA5", "Output: one Zarr directory per year")
    log("ERA5", f"Free disk: {human_bytes(free_space())}")

    opened = time.time()

    filesystem = gcsfs.GCSFileSystem(
        token="anon",
        retries=10,
    )

    dataset = xr.open_zarr(
        filesystem.get_mapper(ERA5_SOURCE),
        consolidated=True,
        chunks={},
    )

    log(
        "ERA5",
        f"Metadata opened in {time.time() - opened:.1f} s",
    )

    missing = [
        name
        for name in ERA5_VARIABLES
        if name not in dataset.data_vars
    ]

    if missing:
        raise RuntimeError(
            "Missing ERA5 variables: " + ", ".join(missing)
        )

    total_years = 2022 - 1979 + 1

    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE,
    )

    for index, year in enumerate(
        range(1979, 2023),
        start=1,
    ):
        destination = ERA5_DIR / f"era5_{year}.zarr"
        partial = ERA5_DIR / f".era5_{year}.partial.zarr"
        success = destination / "_SUCCESS"

        if success.exists():
            log(
                "ERA5",
                f"{year}: already complete "
                f"({index}/{total_years}) | "
                f"{human_bytes(directory_size(destination))}",
            )
            continue

        if partial.exists():
            log(
                "ERA5",
                f"{year}: removing incomplete previous attempt",
            )
            shutil.rmtree(partial)

        if destination.exists():
            log(
                "ERA5",
                f"{year}: incomplete final directory found; removing",
            )
            shutil.rmtree(destination)

        yearly = dataset[ERA5_VARIABLES].sel(
            time=slice(
                f"{year}-01-01T00:00:00",
                f"{year}-12-31T18:00:00",
            )
        )

        log("ERA5", "=" * 65)
        log(
            "ERA5",
            f"{year}: START ({index}/{total_years})",
        )
        log(
            "ERA5",
            f"{year}: timestamps={yearly.sizes['time']}",
        )
        log(
            "ERA5",
            f"{year}: uncompressed volume="
            f"{human_bytes(raw_size(yearly))}",
        )
        log(
            "ERA5",
            f"{year}: writing to {partial}",
        )

        yearly = yearly.chunk(
            {
                "time": 8,
                "level": 13,
                "longitude": 240,
                "latitude": 121,
            }
        )

        encoding = {
            name: {"compressor": compressor}
            for name in yearly.data_vars
        }

        started = time.time()

        with dask.config.set(
            scheduler="threads",
            num_workers=4,
        ):
            yearly.to_zarr(
                partial,
                mode="w",
                consolidated=True,
                encoding=encoding,
            )

        (partial / "_SUCCESS").touch()
        partial.rename(destination)

        elapsed = time.time() - started

        log(
            "ERA5",
            f"{year}: COMPLETE | "
            f"local={human_bytes(directory_size(destination))} | "
            f"elapsed={elapsed / 3600:.2f} h | "
            f"free={human_bytes(free_space())}",
        )

        if free_space() < 20 * 1024**3:
            raise RuntimeError(
                "Less than 20 GiB free; stopping ERA5 safely"
            )

    log(
        "ERA5",
        f"ALL YEARS COMPLETE | "
        f"total={human_bytes(directory_size(ERA5_DIR))}",
    )


def worker(name, function):
    try:
        function()
        log(name, "WORKER FINISHED SUCCESSFULLY")

    except KeyboardInterrupt:
        log(name, "Interrupted")

    except Exception:
        log(name, "WORKER FAILED")
        log(name, traceback.format_exc())
        raise


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    log("LAUNCHER", "=" * 70)
    log("LAUNCHER", "Public ConserveFM dataset downloader")
    log("LAUNCHER", "No accounts, tokens or API keys")
    log("LAUNCHER", "Starting three datasets in parallel")
    log("LAUNCHER", f"Free disk: {human_bytes(free_space())}")
    log("LAUNCHER", "=" * 70)

    context = mp.get_context("spawn")

    jobs = {
        "ERA5": download_era5,
        "GHCN": download_ghcn,
        "WEATHERREAL": download_weatherreal,
    }

    processes = {}

    for name, function in jobs.items():
        process = context.Process(
            target=worker,
            args=(name, function),
            name=f"ConserveFM-{name}",
        )

        process.start()
        processes[name] = process

        log(
            "LAUNCHER",
            f"Started {name}, PID={process.pid}",
        )

    try:
        while any(
            process.is_alive()
            for process in processes.values()
        ):
            time.sleep(60)

            status = []

            for name, process in processes.items():
                if process.is_alive():
                    state = f"RUNNING pid={process.pid}"
                else:
                    state = f"DONE code={process.exitcode}"

                status.append(f"{name}={state}")

            log(
                "STATUS",
                " | ".join(status),
            )

            log(
                "STATUS",
                f"ERA5={human_bytes(directory_size(ERA5_DIR))} | "
                f"GHCN={human_bytes(directory_size(GHCN_DIR))} | "
                f"WeatherReal="
                f"{human_bytes(directory_size(WEATHERREAL_DIR))} | "
                f"free={human_bytes(free_space())}",
            )

    except KeyboardInterrupt:
        log(
            "LAUNCHER",
            "Ctrl+C received; stopping dataset workers",
        )

        for process in processes.values():
            if process.is_alive():
                process.terminate()

        for process in processes.values():
            process.join(timeout=20)

        log(
            "LAUNCHER",
            "Stopped. Existing files will be reused on restart.",
        )

        return

    for process in processes.values():
        process.join()

    log("LAUNCHER", "=" * 70)

    for name, process in processes.items():
        log(
            "LAUNCHER",
            f"{name}: exit code {process.exitcode}",
        )

    log(
        "LAUNCHER",
        f"ERA5 total: {human_bytes(directory_size(ERA5_DIR))}",
    )
    log(
        "LAUNCHER",
        f"GHCN total: {human_bytes(directory_size(GHCN_DIR))}",
    )
    log(
        "LAUNCHER",
        f"WeatherReal total: "
        f"{human_bytes(directory_size(WEATHERREAL_DIR))}",
    )
    log(
        "LAUNCHER",
        f"Free disk: {human_bytes(free_space())}",
    )
    log("LAUNCHER", "=" * 70)


if __name__ == "__main__":
    main()
