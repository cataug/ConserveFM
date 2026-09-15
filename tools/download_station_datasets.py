#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import os
import re
import shutil
import threading
import time
from datetime import datetime


ROOT = Path.home() / "ConserveFM"
DATA = ROOT / "data"

GHCN_DIR = DATA / "GHCN_Daily_by_year"
ISD_DIR = DATA / "ISD_2023_stations"

GHCN_BASE = (
    "https://www.ncei.noaa.gov/pub/data/ghcn/"
    "daily/by_year"
)

ISD_BASE = (
    "https://www.ncei.noaa.gov/data/"
    "global-hourly/access/2023"
)

PRINT_LOCK = threading.Lock()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(label, message):
    with PRINT_LOCK:
        print(
            f"{now()} [{label:<8}] {message}",
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
    total = 0

    if not path.exists():
        return 0

    for root, _, files in os.walk(path):
        for filename in files:
            candidate = Path(root) / filename

            try:
                total += candidate.stat().st_size
            except OSError:
                pass

    return total


def download_file(
    url,
    destination,
    label,
    retries=50,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() and destination.stat().st_size > 0:
        return "skipped", destination.stat().st_size

    partial = Path(str(destination) + ".part")

    for attempt in range(1, retries + 1):
        existing = (
            partial.stat().st_size
            if partial.exists()
            else 0
        )

        headers = {
            "User-Agent": "Mozilla/5.0 ConserveFM downloader",
            "Accept": "*/*",
        }

        if existing:
            headers["Range"] = f"bytes={existing}-"

        request = Request(
            url,
            headers=headers,
        )

        try:
            response = urlopen(
                request,
                timeout=120,
            )

            status = getattr(response, "status", 200)

            if existing and status == 206:
                mode = "ab"
                downloaded = existing
            else:
                mode = "wb"
                downloaded = 0

            content_range = response.headers.get(
                "Content-Range"
            )
            content_length = response.headers.get(
                "Content-Length"
            )

            total = None

            if content_range and "/" in content_range:
                final_value = content_range.rsplit(
                    "/",
                    1,
                )[-1]

                if final_value.isdigit():
                    total = int(final_value)

            elif content_length and content_length.isdigit():
                total = (
                    downloaded
                    + int(content_length)
                )

            with partial.open(mode) as output:
                while True:
                    block = response.read(
                        1024 * 1024
                    )

                    if not block:
                        break

                    output.write(block)
                    downloaded += len(block)

            if total is not None and downloaded < total:
                raise RuntimeError(
                    f"incomplete response: "
                    f"{downloaded}/{total} bytes"
                )

            os.replace(
                partial,
                destination,
            )

            return "downloaded", destination.stat().st_size

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            RuntimeError,
            OSError,
        ) as error:
            delay = min(
                60,
                3 + attempt * 2,
            )

            log(
                label,
                (
                    f"retry {attempt}/{retries}: "
                    f"{destination.name} | "
                    f"{type(error).__name__}: {error} | "
                    f"saved={human_bytes(existing)}"
                ),
            )

            time.sleep(delay)

    raise RuntimeError(
        f"failed after {retries} attempts: {url}"
    )


def download_ghcn():
    GHCN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    years = list(range(1979, 2024))

    log(
        "GHCN",
        (
            f"Starting {len(years)} yearly files, "
            f"1979–2023, workers=4"
        ),
    )

    started = time.time()
    completed = 0
    downloaded_bytes = 0

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:
        futures = {}

        for year in years:
            url = f"{GHCN_BASE}/{year}.csv.gz"
            destination = (
                GHCN_DIR
                / f"{year}.csv.gz"
            )

            future = executor.submit(
                download_file,
                url,
                destination,
                "GHCN",
            )

            futures[future] = year

        for future in as_completed(futures):
            year = futures[future]

            try:
                state, size = future.result()
                completed += 1
                downloaded_bytes += size

                log(
                    "GHCN",
                    (
                        f"{year}: {state} | "
                        f"{human_bytes(size)} | "
                        f"{completed}/{len(years)}"
                    ),
                )

            except Exception as error:
                log(
                    "GHCN",
                    f"{year}: FAILED: {error}",
                )

                raise

    elapsed = time.time() - started

    log(
        "GHCN",
        (
            f"COMPLETE | files={completed} | "
            f"size={human_bytes(directory_size(GHCN_DIR))} | "
            f"elapsed={elapsed / 3600:.2f} h"
        ),
    )


def fetch_isd_file_list():
    log(
        "ISD",
        f"Reading official station index: {ISD_BASE}/",
    )

    for attempt in range(1, 31):
        try:
            request = Request(
                ISD_BASE + "/",
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "ConserveFM downloader"
                    )
                },
            )

            with urlopen(
                request,
                timeout=120,
            ) as response:
                html = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

            files = sorted(
                set(
                    re.findall(
                        r'href="([0-9]{11}\.csv)"',
                        html,
                    )
                )
            )

            if not files:
                raise RuntimeError(
                    "no station CSV links found"
                )

            return files

        except Exception as error:
            log(
                "ISD",
                (
                    f"index retry {attempt}/30: "
                    f"{error}"
                ),
            )

            time.sleep(min(60, attempt * 3))

    raise RuntimeError(
        "could not read ISD station index"
    )


def download_isd():
    ISD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = fetch_isd_file_list()

    log(
        "ISD",
        (
            f"Found {len(files):,} station CSV files | "
            f"year=2023 | workers=8"
        ),
    )

    started = time.time()
    completed = 0
    failed = 0
    bytes_done = 0

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        futures = {}

        for filename in files:
            url = f"{ISD_BASE}/{filename}"
            destination = ISD_DIR / filename

            future = executor.submit(
                download_file,
                url,
                destination,
                "ISD",
            )

            futures[future] = filename

        for future in as_completed(futures):
            filename = futures[future]

            try:
                _, size = future.result()
                bytes_done += size

            except Exception as error:
                failed += 1

                log(
                    "ISD",
                    f"{filename}: FAILED: {error}",
                )

            completed += 1

            if (
                completed % 100 == 0
                or completed == len(files)
            ):
                elapsed = max(
                    time.time() - started,
                    0.001,
                )

                log(
                    "ISD",
                    (
                        f"progress={completed:,}/"
                        f"{len(files):,} | "
                        f"failed={failed} | "
                        f"local={human_bytes(bytes_done)} | "
                        f"average="
                        f"{human_bytes(bytes_done / elapsed)}/s"
                    ),
                )

    log(
        "ISD",
        (
            f"COMPLETE | files={completed - failed:,} | "
            f"failed={failed} | "
            f"size={human_bytes(directory_size(ISD_DIR))}"
        ),
    )

    if failed:
        raise RuntimeError(
            f"{failed} ISD files failed; "
            f"rerun will retry only missing files"
        )


def main():
    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "MAIN",
        "Starting GHCN and ISD simultaneously",
    )
    log(
        "MAIN",
        (
            f"Free disk: "
            f"{human_bytes(shutil.disk_usage(DATA).free)}"
        ),
    )

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        ghcn_future = executor.submit(
            download_ghcn
        )
        isd_future = executor.submit(
            download_isd
        )

        while not (
            ghcn_future.done()
            and isd_future.done()
        ):
            log(
                "STATUS",
                (
                    f"GHCN={human_bytes(directory_size(GHCN_DIR))} | "
                    f"ISD={human_bytes(directory_size(ISD_DIR))} | "
                    f"free={human_bytes(shutil.disk_usage(DATA).free)}"
                ),
            )

            time.sleep(60)

        ghcn_future.result()
        isd_future.result()

    log(
        "MAIN",
        "BOTH STATION DATASETS COMPLETE",
    )


if __name__ == "__main__":
    main()
