#!/usr/bin/env python3

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path.home() / "ConserveFM"

DATASET = (
    ROOT
    / "data/ERA5_WeatherBench2_1979_2022"
)

PYTHON = ROOT / ".venv_data/bin/python"

DOWNLOADER = (
    ROOT
    / "tools/download_era5_parallel_raw.py"
)

LOG_DIR = ROOT / "logs"

WORKERS = int(
    os.environ.get("ERA5_WORKERS", "32")
)

STALL_MINUTES = int(
    os.environ.get("ERA5_STALL_MINUTES", "10")
)

CHECK_SECONDS = 60
RESTART_DELAY_SECONDS = 10


def now() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def human_bytes(value: int | float) -> str:
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


def read_write_bytes(pid: int) -> int | None:
    """
    Возвращает реальные байты, записанные процессом на диск.
    Не считает его stdout и строки прогресса.
    """
    path = Path(f"/proc/{pid}/io")

    try:
        values = {}

        for line in path.read_text().splitlines():
            key, value = line.split(":", 1)
            values[key.strip()] = int(value.strip())

        return values.get("write_bytes", 0)

    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        return None


def dataset_size() -> str:
    result = subprocess.run(
        ["du", "-sh", str(DATASET)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    if result.returncode == 0:
        return result.stdout.strip()

    return "size unavailable"


def clean_partial_files() -> None:
    removed = 0
    removed_bytes = 0

    if not DATASET.exists():
        return

    for partial in DATASET.rglob("*.part"):
        try:
            removed_bytes += partial.stat().st_size
            partial.unlink()
            removed += 1
        except OSError:
            pass

    print(
        f"{now()} [WATCHDOG] "
        f"removed partial files: {removed} "
        f"({human_bytes(removed_bytes)})",
        flush=True,
    )


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    print(
        f"{now()} [WATCHDOG] "
        f"stopping process group {process.pid}",
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
        f"{now()} [WATCHDOG] "
        "SIGTERM did not work; sending SIGKILL",
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


def copy_output(
    process: subprocess.Popen,
    log_file,
) -> None:
    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
            flush=True,
        )

        log_file.write(line)
        log_file.flush()


def run_cycle(cycle: int) -> str:
    clean_partial_files()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        LOG_DIR
        / f"era5_stall_cycle_{cycle:03d}_{stamp}.log"
    )

    environment = os.environ.copy()
    environment["ERA5_WORKERS"] = str(WORKERS)

    command = [
        str(PYTHON),
        "-u",
        str(DOWNLOADER),
    ]

    print()
    print("=" * 76)
    print(
        f"{now()} [WATCHDOG] cycle {cycle}"
    )
    print(
        f"{now()} [WATCHDOG] workers: {WORKERS}"
    )
    print(
        f"{now()} [WATCHDOG] restart after "
        f"{STALL_MINUTES} minutes without disk writes"
    )
    print(
        f"{now()} [WATCHDOG] current {dataset_size()}"
    )
    print(
        f"{now()} [WATCHDOG] log: {log_path}"
    )
    print("=" * 76, flush=True)

    log_file = log_path.open(
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
        target=copy_output,
        args=(process, log_file),
        daemon=True,
    )

    reader.start()

    started = time.time()
    last_change_time = started

    previous_write_bytes = (
        read_write_bytes(process.pid) or 0
    )

    status_counter = 0

    try:
        while process.poll() is None:
            time.sleep(CHECK_SECONDS)

            current_time = time.time()
            current_write_bytes = read_write_bytes(
                process.pid
            )

            if current_write_bytes is None:
                break

            written_during_minute = max(
                0,
                current_write_bytes
                - previous_write_bytes,
            )

            if written_during_minute > 0:
                last_change_time = current_time

            previous_write_bytes = current_write_bytes

            idle_seconds = (
                current_time - last_change_time
            )

            status_counter += 1

            print(
                f"{now()} [WATCHDOG] "
                f"pid={process.pid} | "
                f"written_last_minute="
                f"{human_bytes(written_during_minute)} | "
                f"no_disk_writes="
                f"{idle_seconds / 60:.1f} min",
                flush=True,
            )

            # Размер печатаем раз в пять минут, чтобы du
            # не обходил десятки тысяч файлов каждую минуту.
            if status_counter % 5 == 0:
                print(
                    f"{now()} [WATCHDOG] "
                    f"{dataset_size()}",
                    flush=True,
                )

            if idle_seconds >= STALL_MINUTES * 60:
                print(
                    f"{now()} [WATCHDOG] "
                    f"NO DISK WRITES FOR "
                    f"{idle_seconds / 60:.1f} MINUTES — "
                    "RESTARTING",
                    flush=True,
                )

                stop_process(process)
                reader.join(timeout=5)
                log_file.close()

                clean_partial_files()

                return "stalled"

        return_code = process.wait()

        reader.join(timeout=5)
        log_file.close()

        if (
            return_code == 0
            and (DATASET / "_SUCCESS").exists()
        ):
            return "complete"

        print(
            f"{now()} [WATCHDOG] "
            f"downloader exited with code "
            f"{return_code}; restarting",
            flush=True,
        )

        return "failed"

    except KeyboardInterrupt:
        print(
            f"\n{now()} [WATCHDOG] "
            "Ctrl+C received — stopping without restart",
            flush=True,
        )

        stop_process(process)
        reader.join(timeout=5)
        log_file.close()

        clean_partial_files()

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
                f"{now()} [WATCHDOG] "
                "stopped by user; SSH remains open",
                flush=True,
            )
            return

        cycle += 1

        print(
            f"{now()} [WATCHDOG] "
            f"restarting in "
            f"{RESTART_DELAY_SECONDS} seconds...",
            flush=True,
        )

        time.sleep(RESTART_DELAY_SECONDS)

    print(
        f"{now()} [WATCHDOG] "
        "_SUCCESS already exists; nothing to do",
        flush=True,
    )


if __name__ == "__main__":
    main()
