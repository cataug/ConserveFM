#!/usr/bin/env python3

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path


ROOT = Path.home() / "ConserveFM"

DATASET = (
    ROOT
    / "data/ERA5_WeatherBench2_1979_2022"
)

PYTHON = (
    ROOT
    / ".venv_data/bin/python"
)

DOWNLOADER = (
    ROOT
    / "tools/download_era5_parallel_raw.py"
)

LOG_DIR = ROOT / "logs"

WORKERS = int(
    os.environ.get("ERA5_WORKERS", "32")
)

# Первые минуты загрузчик читает metadata и проверяет файлы.
GRACE_SECONDS = int(
    os.environ.get("ERA5_GRACE_MINUTES", "20")
) * 60

# Скорость оценивается по этому окну.
SPEED_WINDOW_SECONDS = int(
    os.environ.get("ERA5_SPEED_WINDOW_MINUTES", "10")
) * 60

# Ниже этой средней скорости процесс перезапускается.
MIN_SPEED = float(
    os.environ.get("ERA5_MIN_MIB_S", "1.0")
) * 1024 * 1024

# Полное отсутствие роста.
STALL_SECONDS = int(
    os.environ.get("ERA5_STALL_MINUTES", "20")
) * 60

CHECK_INTERVAL = 60


def now() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def human_bytes(value: float) -> str:
    value = float(value)

    for unit in [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]:
        if value < 1024:
            return f"{value:,.2f} {unit}"

        value /= 1024

    return f"{value:,.2f} PiB"


def directory_size(path: Path) -> int:
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


def clean_partial_files() -> None:
    removed = 0
    removed_bytes = 0

    for partial in DATASET.rglob("*.part"):
        try:
            removed_bytes += partial.stat().st_size
            partial.unlink()
            removed += 1
        except OSError:
            pass

    print(
        f"{now()} [WATCHDOG] cleaned partial files: "
        f"{removed}, {human_bytes(removed_bytes)}",
        flush=True,
    )


def stop_process(
    process: subprocess.Popen,
) -> None:
    if process.poll() is not None:
        return

    print(
        f"{now()} [WATCHDOG] sending SIGTERM "
        f"to process group {process.pid}",
        flush=True,
    )

    try:
        os.killpg(
            process.pid,
            signal.SIGTERM,
        )
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        pass

    print(
        f"{now()} [WATCHDOG] process did not stop; "
        "sending SIGKILL",
        flush=True,
    )

    try:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def stream_output(
    process: subprocess.Popen,
    log_stream,
) -> None:
    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)

        log_stream.write(line)
        log_stream.flush()


def run_cycle(cycle: int) -> str:
    clean_partial_files()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        LOG_DIR
        / f"era5_smart_cycle_{cycle:03d}_{stamp}.log"
    )

    environment = os.environ.copy()
    environment["ERA5_WORKERS"] = str(WORKERS)

    command = [
        str(PYTHON),
        "-u",
        str(DOWNLOADER),
    ]

    initial_size = directory_size(DATASET)

    print()
    print("=" * 74)
    print(
        f"{now()} [WATCHDOG] ERA5 cycle {cycle}"
    )
    print(
        f"{now()} [WATCHDOG] workers: {WORKERS}"
    )
    print(
        f"{now()} [WATCHDOG] current size: "
        f"{human_bytes(initial_size)}"
    )
    print(
        f"{now()} [WATCHDOG] minimum acceptable "
        f"rolling speed: "
        f"{MIN_SPEED / 1024 / 1024:.2f} MiB/s"
    )
    print(
        f"{now()} [WATCHDOG] log: {log_path}"
    )
    print("=" * 74, flush=True)

    log_stream = log_path.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    reader = threading.Thread(
        target=stream_output,
        args=(process, log_stream),
        daemon=True,
    )

    reader.start()

    started = time.time()
    history: deque[tuple[float, int]] = deque()
    last_growth_time = started
    previous_size = initial_size
    low_speed_checks = 0

    try:
        while process.poll() is None:
            time.sleep(CHECK_INTERVAL)

            current_time = time.time()
            current_size = directory_size(DATASET)

            if current_size > previous_size:
                last_growth_time = current_time

            previous_size = current_size

            history.append(
                (current_time, current_size)
            )

            while (
                history
                and current_time - history[0][0]
                > SPEED_WINDOW_SECONDS
            ):
                history.popleft()

            elapsed = current_time - started
            no_growth = (
                current_time - last_growth_time
            )

            rolling_speed = 0.0

            if (
                len(history) >= 2
                and history[-1][0] > history[0][0]
            ):
                rolling_speed = (
                    history[-1][1] - history[0][1]
                ) / (
                    history[-1][0] - history[0][0]
                )

            print(
                f"{now()} [WATCHDOG] "
                f"size={human_bytes(current_size)} | "
                f"cycle_added="
                f"{human_bytes(current_size - initial_size)} | "
                f"rolling="
                f"{human_bytes(rolling_speed)}/s | "
                f"no_growth={no_growth / 60:.1f} min | "
                f"pid={process.pid}",
                flush=True,
            )

            if elapsed < GRACE_SECONDS:
                continue

            if no_growth >= STALL_SECONDS:
                print(
                    f"{now()} [WATCHDOG] no growth for "
                    f"{no_growth / 60:.1f} minutes; "
                    "restarting",
                    flush=True,
                )

                stop_process(process)
                return "stall"

            enough_history = (
                history
                and history[-1][0] - history[0][0]
                >= SPEED_WINDOW_SECONDS * 0.8
            )

            if (
                enough_history
                and rolling_speed < MIN_SPEED
            ):
                low_speed_checks += 1

                print(
                    f"{now()} [WATCHDOG] low-speed "
                    f"check {low_speed_checks}/3",
                    flush=True,
                )
            else:
                low_speed_checks = 0

            if low_speed_checks >= 3:
                print(
                    f"{now()} [WATCHDOG] rolling speed "
                    f"{human_bytes(rolling_speed)}/s "
                    "is persistently too low; restarting",
                    flush=True,
                )

                stop_process(process)
                return "slow"

        return_code = process.wait()

        reader.join(timeout=5)
        log_stream.close()

        if (
            return_code == 0
            and (DATASET / "_SUCCESS").exists()
        ):
            return "complete"

        print(
            f"{now()} [WATCHDOG] downloader exited "
            f"with code {return_code}; restarting",
            flush=True,
        )

        return "failed"

    except KeyboardInterrupt:
        print(
            f"\n{now()} [WATCHDOG] Ctrl+C received",
            flush=True,
        )

        stop_process(process)
        reader.join(timeout=5)
        log_stream.close()

        return "user_stop"


def main() -> None:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycle = 1

    while not (DATASET / "_SUCCESS").exists():
        result = run_cycle(cycle)

        if result == "complete":
            print(
                f"{now()} [WATCHDOG] "
                "ERA5 DOWNLOAD COMPLETE",
                flush=True,
            )
            return

        if result == "user_stop":
            print(
                f"{now()} [WATCHDOG] stopped by user; "
                "SSH shell remains open",
                flush=True,
            )
            return

        cycle += 1

        print(
            f"{now()} [WATCHDOG] restarting in "
            "15 seconds...",
            flush=True,
        )

        time.sleep(15)

    print(
        f"{now()} [WATCHDOG] _SUCCESS already exists",
        flush=True,
    )


if __name__ == "__main__":
    main()
