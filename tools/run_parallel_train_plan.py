#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path.home() / "ConserveFM"

DEFAULT_PLAN = (
    ROOT
    / "manifests/experiments_v1/planned_runs.csv"
)

DEFAULT_RUN_ROOT = ROOT / "runs"

TRAIN_ENTRY = ROOT / "tools/train_conservefm.py"

RESEARCH_MANIFEST = (
    ROOT / "manifests/research_v1"
)

ERA5 = (
    ROOT
    / "data/ERA5_WeatherBench2_1979_2022"
)

FORECAST_ROOT = ROOT / "forecasts"

STATS = (
    RESEARCH_MANIFEST
    / "era5_channel_stats.npz"
)


def now():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
    )

    p.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )

    p.add_argument(
        "--python",
        required=True,
    )

    p.add_argument(
        "--data-site",
        required=True,
    )

    p.add_argument(
        "--gpu",
        default="0",
    )

    p.add_argument(
        "--max-parallel",
        type=int,
        default=4,
    )

    p.add_argument(
        "--reserve-per-job-gib",
        type=float,
        default=7.0,
    )

    p.add_argument(
        "--headroom-gib",
        type=float,
        default=4.0,
    )

    p.add_argument(
        "--poll-seconds",
        type=int,
        default=10,
    )

    p.add_argument(
        "--launch-gap",
        type=int,
        default=15,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    p.add_argument(
        "--backbone",
        action="append",
        default=[],
    )

    p.add_argument(
        "--method",
        action="append",
        default=[],
    )

    p.add_argument(
        "--retry-failed",
        action="store_true",
    )

    p.add_argument(
        "--smoke",
        action="store_true",
    )

    p.add_argument(
        "--smoke-train-examples",
        type=int,
        default=4,
    )

    p.add_argument(
        "--smoke-val-examples",
        type=int,
        default=2,
    )

    return p.parse_args()


def load_plan(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    managed = [
        "status",
        "result_path",
        "started_at",
        "finished_at",
        "exit_code",
        "attempt",
        "last_error",
    ]

    for field in managed:
        if field not in fields:
            fields.append(field)

    for row in rows:
        for field in managed:
            row.setdefault(field, "")

        if not row["status"]:
            row["status"] = "planned"

        if not row["attempt"]:
            row["attempt"] = "0"

    return rows, fields


def save_plan(path, rows, fields):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def gpu_status(gpu):
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
    ).strip()

    total, used, free, util = [
        float(x.strip())
        for x in out.split(",")
    ]

    return {
        "total_mib": total,
        "used_mib": used,
        "free_mib": free,
        "util": util,
    }


def human_gib(mib):
    return f"{mib / 1024:.1f} GiB"


def stream_output(
    run_id,
    process,
    log_path,
):
    assert process.stdout is not None

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as log:
        for line in process.stdout:
            log.write(line)
            log.flush()

            print(
                f"[{run_id}] {line}",
                end="",
                flush=True,
            )


def build_command(
    row,
    args,
    run_dir,
):
    cmd = [
        args.python,
        "-u",
        str(TRAIN_ENTRY),

        "--run-id",
        row["run_id"],

        "--backbone",
        row["backbone"],

        "--family",
        row["family"],

        "--method",
        row["method"],

        "--seed",
        str(row["seed"]),

        "--research-manifest",
        str(RESEARCH_MANIFEST),

        "--era5-zarr",
        str(ERA5),

        "--forecast-root",
        str(FORECAST_ROOT),

        "--stats-path",
        str(STATS),

        "--output-dir",
        str(run_dir),
    ]

    if args.smoke:
        cmd += [
            "--smoke",
            "--max-train-examples",
            str(args.smoke_train_examples),
            "--max-val-examples",
            str(args.smoke_val_examples),
            "--stats-samples",
            "1",
            "--width",
            "8",
            "--depth",
            "1",
        ]

    return cmd


def forecast_ready(backbone):
    store = (
        FORECAST_ROOT
        / f"{backbone}.zarr"
    )

    return store.exists()


def eligible(row, args):
    if row.get("job_type") != "train":
        return False

    if (
        args.backbone
        and row["backbone"]
        not in args.backbone
    ):
        return False

    if (
        args.method
        and row["method"]
        not in args.method
    ):
        return False

    status = (
        row.get("status") or "planned"
    )

    if status in {
        "planned",
        "pending",
        "interrupted",
    }:
        return True

    if (
        status == "failed"
        and args.retry_failed
    ):
        return True

    return False


def stop_process(process):
    if process.poll() is not None:
        return

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

    try:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )
    except ProcessLookupError:
        pass


def main():
    args = parse_args()

    args.plan = (
        args.plan.expanduser().resolve()
    )

    args.run_root = (
        args.run_root.expanduser().resolve()
    )

    args.run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not STATS.exists():
        raise RuntimeError(
            f"Missing normalization stats: {STATS}"
        )

    rows, fields = load_plan(
        args.plan
    )

    #
    # Recover jobs whose old scheduler died.
    #
    changed = False

    for row in rows:
        if row.get("status") == "running":
            row["status"] = "interrupted"
            row["last_error"] = (
                "Recovered stale running job"
            )
            changed = True

    if changed:
        save_plan(
            args.plan,
            rows,
            fields,
        )

    candidates = [
        row
        for row in rows
        if eligible(row, args)
    ]

    if args.limit > 0:
        candidates = candidates[
            :args.limit
        ]

    #
    # In full mode, never silently use persistence.
    #
    if not args.smoke:
        missing = sorted({
            row["backbone"]
            for row in candidates
            if not forecast_ready(
                row["backbone"]
            )
        })

        if missing:
            print(
                "[PREFLIGHT] Missing full forecast stores:",
                flush=True,
            )

            for backbone in missing:
                print(
                    "  ",
                    FORECAST_ROOT
                    / f"{backbone}.zarr",
                    flush=True,
                )

            print(
                "\nRefusing to launch invalid full runs.",
                flush=True,
            )

            return 2

    print("=" * 78)
    print("ConserveFM GPU-aware parallel scheduler")
    print("=" * 78)
    print("plan             :", args.plan)
    print("candidate jobs   :", len(candidates))
    print("GPU              :", args.gpu)
    print("max parallel     :", args.max_parallel)
    print(
        "reserve/job      :",
        f"{args.reserve_per_job_gib:.1f} GiB",
    )
    print(
        "GPU headroom     :",
        f"{args.headroom_gib:.1f} GiB",
    )
    print("smoke            :", args.smoke)
    print("=" * 78)

    if not candidates:
        print("Nothing to run.")
        return 0

    by_id = {
        row["run_id"]: row
        for row in rows
    }

    pending = list(candidates)

    running = {}

    completed = 0
    failed = 0

    last_launch = 0.0

    stop_requested = False

    try:
        while pending or running:
            #
            # Harvest finished jobs.
            #
            finished = []

            for run_id, info in list(
                running.items()
            ):
                process = info["process"]

                code = process.poll()

                if code is None:
                    continue

                info["thread"].join(
                    timeout=5
                )

                row = by_id[run_id]

                row["finished_at"] = now()
                row["exit_code"] = str(code)
                row["result_path"] = str(
                    info["run_dir"]
                )

                result_json = (
                    info["run_dir"]
                    / "result.json"
                )

                if (
                    code == 0
                    and result_json.exists()
                ):
                    row["status"] = "completed"
                    row["last_error"] = ""
                    completed += 1

                    print(
                        f"\n[DONE] {run_id}",
                        flush=True,
                    )

                else:
                    row["status"] = "failed"
                    row["last_error"] = (
                        f"exit_code={code}"
                    )
                    failed += 1

                    print(
                        f"\n[FAILED] {run_id} "
                        f"exit={code}",
                        flush=True,
                    )

                save_plan(
                    args.plan,
                    rows,
                    fields,
                )

                finished.append(
                    run_id
                )

            for run_id in finished:
                del running[run_id]

            #
            # GPU state.
            #
            gpu = gpu_status(
                args.gpu
            )

            free_gib = (
                gpu["free_mib"] / 1024
            )

            memory_slots = max(
                0,
                math.floor(
                    (
                        free_gib
                        - args.headroom_gib
                    )
                    / args.reserve_per_job_gib
                ),
            )

            available_slots = min(
                args.max_parallel
                - len(running),
                memory_slots,
            )

            #
            # Stagger launches so nvidia-smi has time
            # to see memory allocated by the previous job.
            #
            if (
                pending
                and available_slots > 0
                and (
                    time.time()
                    - last_launch
                    >= args.launch_gap
                )
            ):
                row = pending.pop(0)

                run_id = row["run_id"]

                run_dir = (
                    args.run_root
                    / run_id
                )

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                log_path = (
                    run_dir
                    / "stdout.log"
                )

                cmd = build_command(
                    row,
                    args,
                    run_dir,
                )

                env = os.environ.copy()

                env["CUDA_VISIBLE_DEVICES"] = (
                    str(args.gpu)
                )

                env["PYTHONUNBUFFERED"] = "1"

                old_pp = env.get(
                    "PYTHONPATH",
                    "",
                )

                env["PYTHONPATH"] = ":".join(
                    x
                    for x in [
                        str(ROOT),
                        args.data_site,
                        old_pp,
                    ]
                    if x
                )

                row["status"] = "running"
                row["started_at"] = now()
                row["finished_at"] = ""
                row["exit_code"] = ""
                row["last_error"] = ""

                row["attempt"] = str(
                    int(
                        row.get(
                            "attempt",
                            "0",
                        )
                        or 0
                    )
                    + 1
                )

                save_plan(
                    args.plan,
                    rows,
                    fields,
                )

                print()
                print(
                    f"[LAUNCH] {run_id}",
                    flush=True,
                )
                print(
                    f"         backbone="
                    f"{row['backbone']} "
                    f"method={row['method']} "
                    f"seed={row['seed']}",
                    flush=True,
                )
                print(
                    f"         GPU free="
                    f"{human_gib(gpu['free_mib'])}",
                    flush=True,
                )

                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )

                thread = threading.Thread(
                    target=stream_output,
                    args=(
                        run_id,
                        process,
                        log_path,
                    ),
                    daemon=True,
                )

                thread.start()

                running[run_id] = {
                    "process": process,
                    "thread": thread,
                    "run_dir": run_dir,
                }

                last_launch = time.time()

            print(
                f"\r[STATUS {now()}] "
                f"running={len(running)} "
                f"queued={len(pending)} "
                f"done={completed} "
                f"failed={failed} | "
                f"GPU used={human_gib(gpu['used_mib'])} "
                f"free={human_gib(gpu['free_mib'])} "
                f"util={gpu['util']:.0f}% "
                f"mem_slots={memory_slots}    ",
                end="",
                flush=True,
            )

            time.sleep(
                args.poll_seconds
            )

    except KeyboardInterrupt:
        print(
            "\n\n[CTRL+C] stopping all running jobs...",
            flush=True,
        )

        stop_requested = True

        for run_id, info in list(
            running.items()
        ):
            stop_process(
                info["process"]
            )

            row = by_id[run_id]
            row["status"] = "interrupted"
            row["finished_at"] = now()
            row["last_error"] = (
                "Interrupted by user"
            )

        save_plan(
            args.plan,
            rows,
            fields,
        )

    print()
    print("=" * 78)
    print("PARALLEL SCHEDULER FINISHED")
    print("completed this session :", completed)
    print("failed this session    :", failed)
    print("still queued           :", len(pending))
    print("=" * 78)

    return (
        130
        if stop_requested
        else (1 if failed else 0)
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
