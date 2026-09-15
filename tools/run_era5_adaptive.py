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

CANDIDATES = [
    int(value)
    for value in os.environ.get(
        "ERA5_WORKER_CANDIDATES",
        "32,48,64,24",
    ).split(",")
    if value.strip()
]

STALL_SECONDS = (
    int(
        os.environ.get(
            "ERA5_STALL_MINUTES",
            "5",
        )
    )
    * 60
)

PROBE_SECONDS = (
    int(
        os.environ.get(
            "ERA5_PROBE_MINUTES",
            "5",
        )
    )
    * 60
)

CHECK_SECONDS = 30

RESTART_DELAY_SECONDS = 10

# Простое число, чтобы каждый рестарт начинал
# с другой части последовательной очереди.
OFFSET_STEP = 997


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
        if abs(value) < 1024:
            return f"{value:,.2f} {unit}"

        value /= 1024

    return f"{value:,.2f} PiB"


def read_write_bytes(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/io")

    try:
        for line in path.read_text().splitlines():
            if line.startswith("write_bytes:"):
                return int(line.split(":", 1)[1])

    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        ValueError,
    ):
        return None

    return 0


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
        f"{now()} [ADAPTIVE] cleaned "
        f"{removed} partial files "
        f"({human_bytes(removed_bytes)})",
        flush=True,
    )


def stop_process(
    process: subprocess.Popen,
) -> None:
    if process.poll() is not None:
        return

    print(
        f"{now()} [ADAPTIVE] stopping "
        f"process group {process.pid}",
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
        f"{now()} [ADAPTIVE] sending SIGKILL",
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
    log_file,
) -> None:
    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()


def run_process(
    workers: int,
    task_offset: int,
    probe: bool,
    run_number: int,
) -> tuple[str, float]:
    clean_partial_files()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    mode = "probe" if probe else "main"

    log_path = (
        LOG_DIR
        / (
            f"era5_adaptive_{mode}_"
            f"w{workers}_{stamp}.log"
        )
    )

    environment = os.environ.copy()
    environment["ERA5_WORKERS"] = str(workers)
    environment["ERA5_TASK_OFFSET"] = str(task_offset)

    print()
    print("=" * 76)
    print(
        f"{now()} [ADAPTIVE] "
        f"{mode.upper()} RUN {run_number}"
    )
    print(
        f"{now()} [ADAPTIVE] workers={workers}"
    )
    print(
        f"{now()} [ADAPTIVE] "
        f"task_offset={task_offset}"
    )
    print(
        f"{now()} [ADAPTIVE] "
        f"stall limit={STALL_SECONDS // 60} min"
    )

    if probe:
        print(
            f"{now()} [ADAPTIVE] "
            f"measurement={PROBE_SECONDS // 60} "
            "minutes after first disk write"
        )

    print(
        f"{now()} [ADAPTIVE] log={log_path}"
    )
    print("=" * 76, flush=True)

    log_file = log_path.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            str(PYTHON),
            "-u",
            str(DOWNLOADER),
        ],
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
        args=(process, log_file),
        daemon=True,
    )

    reader.start()

    started = time.time()
    last_write_time = started

    previous_write = (
        read_write_bytes(process.pid) or 0
    )

    measurement_started = None
    measurement_initial_bytes = None

    try:
        while process.poll() is None:
            time.sleep(CHECK_SECONDS)

            current_time = time.time()
            current_write = read_write_bytes(
                process.pid
            )

            if current_write is None:
                break

            delta = max(
                0,
                current_write - previous_write,
            )

            previous_write = current_write

            if delta > 0:
                last_write_time = current_time

                if measurement_started is None:
                    measurement_started = current_time
                    measurement_initial_bytes = (
                        current_write - delta
                    )

                    print(
                        f"{now()} [ADAPTIVE] "
                        "first disk write detected; "
                        "measurement started",
                        flush=True,
                    )

            idle = current_time - last_write_time

            instant_speed = (
                delta / CHECK_SECONDS
            )

            if measurement_started is not None:
                measured_seconds = (
                    current_time
                    - measurement_started
                )

                measured_bytes = (
                    current_write
                    - measurement_initial_bytes
                )

                average_speed = (
                    measured_bytes
                    / max(measured_seconds, 1)
                )
            else:
                measured_seconds = 0
                measured_bytes = 0
                average_speed = 0

            print(
                f"{now()} [ADAPTIVE] "
                f"workers={workers} | "
                f"last_30s={human_bytes(delta)} | "
                f"instant={human_bytes(instant_speed)}/s | "
                f"measured={human_bytes(measured_bytes)} | "
                f"average={human_bytes(average_speed)}/s | "
                f"idle={idle / 60:.1f} min",
                flush=True,
            )

            if idle >= STALL_SECONDS:
                print(
                    f"{now()} [ADAPTIVE] "
                    f"NO WRITES FOR "
                    f"{idle / 60:.1f} MINUTES",
                    flush=True,
                )

                stop_process(process)
                reader.join(timeout=5)
                log_file.close()

                return "stalled", average_speed

            if (
                probe
                and measurement_started is not None
                and measured_seconds >= PROBE_SECONDS
            ):
                print(
                    f"{now()} [ADAPTIVE] "
                    f"probe complete: workers={workers}, "
                    f"speed={human_bytes(average_speed)}/s",
                    flush=True,
                )

                stop_process(process)
                reader.join(timeout=5)
                log_file.close()

                return "probe_complete", average_speed

        return_code = process.wait()

        reader.join(timeout=5)
        log_file.close()

        if (
            return_code == 0
            and (DATASET / "_SUCCESS").exists()
        ):
            return "complete", average_speed

        return "failed", average_speed

    except KeyboardInterrupt:
        print(
            f"\n{now()} [ADAPTIVE] "
            "Ctrl+C received",
            flush=True,
        )

        stop_process(process)
        reader.join(timeout=5)
        log_file.close()
        clean_partial_files()

        return "user_stop", average_speed


def benchmark_workers(
    offset_counter: int,
) -> tuple[int, int] | tuple[None, int]:
    results: dict[int, float] = {}

    print()
    print("#" * 76)
    print(
        f"{now()} [ADAPTIVE] "
        f"BENCHMARKING WORKERS: {CANDIDATES}"
    )
    print("#" * 76, flush=True)

    run_number = 1

    for workers in CANDIDATES:
        task_offset = (
            offset_counter * OFFSET_STEP
        )

        status, speed = run_process(
            workers=workers,
            task_offset=task_offset,
            probe=True,
            run_number=run_number,
        )

        offset_counter += 1
        run_number += 1

        if status == "complete":
            return None, offset_counter

        if status == "user_stop":
            raise KeyboardInterrupt

        if status == "probe_complete":
            results[workers] = speed
        else:
            results[workers] = 0.0

    print()
    print("=" * 76)
    print(
        f"{now()} [ADAPTIVE] BENCHMARK RESULTS"
    )

    for workers in CANDIDATES:
        print(
            f"  workers={workers:<3} "
            f"speed={human_bytes(results[workers])}/s"
        )

    best_workers = max(
        results,
        key=results.get,
    )

    print(
        f"{now()} [ADAPTIVE] "
        f"SELECTED: {best_workers} workers "
        f"at {human_bytes(results[best_workers])}/s"
    )
    print("=" * 76, flush=True)

    return best_workers, offset_counter


def main() -> None:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    offset_counter = 0
    main_run_number = 1

    try:
        while not (DATASET / "_SUCCESS").exists():
            best_workers, offset_counter = (
                benchmark_workers(offset_counter)
            )

            if best_workers is None:
                break

            task_offset = (
                offset_counter * OFFSET_STEP
            )
            offset_counter += 1

            status, _ = run_process(
                workers=best_workers,
                task_offset=task_offset,
                probe=False,
                run_number=main_run_number,
            )

            main_run_number += 1

            if status == "complete":
                break

            if status == "user_stop":
                return

            print(
                f"{now()} [ADAPTIVE] "
                "main run stalled or failed; "
                "rebenchmarking worker counts in "
                f"{RESTART_DELAY_SECONDS} seconds",
                flush=True,
            )

            time.sleep(
                RESTART_DELAY_SECONDS
            )

    except KeyboardInterrupt:
        print(
            f"{now()} [ADAPTIVE] "
            "stopped by user; SSH remains open",
            flush=True,
        )

        return

    print(
        f"{now()} [ADAPTIVE] "
        "ERA5 DOWNLOAD COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
