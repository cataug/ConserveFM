#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path.home() / "ConserveFM"

DEFAULT_PLAN = (
    ROOT
    / "manifests/experiments_v1/planned_runs.csv"
)

DEFAULT_RUN_ROOT = ROOT / "runs"

DEFAULT_TRAIN_ENTRY = (
    ROOT
    / "tools/train_conservefm.py"
)

DEFAULT_EVAL_ENTRY = (
    ROOT
    / "tools/eval_conservefm.py"
)

DEFAULT_RESEARCH_MANIFEST = (
    ROOT
    / "manifests/research_v1"
)

MANAGED_FIELDS = [
    "status",
    "result_path",
    "started_at",
    "finished_at",
    "exit_code",
    "attempt",
    "last_error",
]


def now():
    return datetime.now().isoformat(
        timespec="seconds"
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
        "--research-manifest",
        type=Path,
        default=DEFAULT_RESEARCH_MANIFEST,
    )

    p.add_argument(
        "--train-entry",
        type=Path,
        default=DEFAULT_TRAIN_ENTRY,
    )

    p.add_argument(
        "--eval-entry",
        type=Path,
        default=DEFAULT_EVAL_ENTRY,
    )

    p.add_argument(
        "--python",
        default=sys.executable,
    )

    p.add_argument(
        "--stage",
        choices=[
            "all",
            "train",
            "eval",
        ],
        default="all",
    )

    p.add_argument(
        "--gpu",
        default="0",
    )

    p.add_argument(
        "--backbone",
        action="append",
        default=[],
        help="Can be repeated",
    )

    p.add_argument(
        "--method",
        action="append",
        default=[],
        help="Can be repeated",
    )

    p.add_argument(
        "--family",
        action="append",
        default=[],
        help="Can be repeated",
    )

    p.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Can be repeated",
    )

    p.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run only specific run_id(s)",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--retry-failed",
        action="store_true",
    )

    p.add_argument(
        "--retry-interrupted",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun even completed rows",
    )

    p.add_argument(
        "--skip-dependency-check",
        action="store_true",
    )

    return p.parse_args()


def load_plan(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Plan not found: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        rows = list(reader)

        fields = list(
            reader.fieldnames or []
        )

    for field in MANAGED_FIELDS:
        if field not in fields:
            fields.append(field)

    for row in rows:
        for field in MANAGED_FIELDS:
            row.setdefault(
                field,
                "",
            )

        if not row["status"]:
            row["status"] = "planned"

        if not row["attempt"]:
            row["attempt"] = "0"

    return rows, fields


def atomic_write_plan(
    path: Path,
    rows,
    fields,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def write_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def row_index(rows):
    return {
        row["run_id"]: row
        for row in rows
    }


def selected(row, args):
    if (
        args.stage != "all"
        and row["job_type"] != args.stage
    ):
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

    if (
        args.family
        and row["family"]
        not in args.family
    ):
        return False

    if (
        args.seed
        and str(row["seed"])
        not in {
            str(x)
            for x in args.seed
        }
    ):
        return False

    if (
        args.run_id
        and row["run_id"]
        not in args.run_id
    ):
        return False

    return True


def should_run(row, args):
    status = (
        row.get(
            "status",
            "planned",
        )
        or "planned"
    )

    if args.force:
        return True

    if status in {
        "planned",
        "pending",
    }:
        return True

    if (
        status == "failed"
        and args.retry_failed
    ):
        return True

    if (
        status == "interrupted"
        and args.retry_interrupted
    ):
        return True

    # A stale running status means
    # previous scheduler process died.
    if status == "running":
        return True

    return False


def dependency_ready(
    row,
    by_id,
    args,
):
    dep = (
        row.get(
            "depends_on",
            "",
        )
        or ""
    ).strip()

    if not dep:
        return True, ""

    if args.skip_dependency_check:
        return True, ""

    dependency = by_id.get(dep)

    if dependency is None:
        return (
            False,
            f"dependency not found: {dep}",
        )

    if (
        dependency.get("status")
        != "completed"
    ):
        return (
            False,
            (
                f"dependency {dep} has "
                f"status={dependency.get('status')}"
            ),
        )

    result_path = (
        dependency.get(
            "result_path",
            "",
        )
        or ""
    ).strip()

    if not result_path:
        return (
            False,
            f"dependency {dep} has no result_path",
        )

    if not Path(result_path).exists():
        return (
            False,
            (
                f"dependency result missing: "
                f"{result_path}"
            ),
        )

    return True, ""


def split_pipe(value):
    value = (
        value or ""
    ).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split("|")
        if x.strip()
    ]


def train_command(
    row,
    args,
    run_dir,
):
    cmd = [
        args.python,
        "-u",
        str(args.train_entry),

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
        str(args.research_manifest),

        "--output-dir",
        str(run_dir),
    ]

    return cmd


def eval_command(
    row,
    args,
    run_dir,
    by_id,
):
    cmd = [
        args.python,
        "-u",
        str(args.eval_entry),

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
        str(args.research_manifest),

        "--output-dir",
        str(run_dir),
    ]

    leads = split_pipe(
        row.get(
            "lead_hours",
            "",
        )
    )

    stress_types = split_pipe(
        row.get(
            "stress_types",
            "",
        )
    )

    severities = split_pipe(
        row.get(
            "severity",
            "",
        )
    )

    if leads:
        cmd += [
            "--lead-hours",
            ",".join(leads),
        ]

    if stress_types:
        cmd += [
            "--stress-types",
            ",".join(
                stress_types
            ),
        ]

    if severities:
        cmd += [
            "--severities",
            ",".join(
                severities
            ),
        ]

    if (
        str(
            row.get(
                "eval_clean",
                "",
            )
        ).lower()
        == "true"
    ):
        cmd.append(
            "--eval-clean"
        )

    if (
        str(
            row.get(
                "eval_stress",
                "",
            )
        ).lower()
        == "true"
    ):
        cmd.append(
            "--eval-stress"
        )

    if (
        str(
            row.get(
                "eval_ghcn",
                "",
            )
        ).lower()
        == "true"
    ):
        cmd.append(
            "--eval-ghcn"
        )

    dep = (
        row.get(
            "depends_on",
            "",
        )
        or ""
    ).strip()

    if dep:
        dep_row = by_id[dep]

        cmd += [
            "--checkpoint-dir",
            dep_row["result_path"],
        ]

    return cmd


def print_command(cmd):
    return " ".join(
        shlex.quote(str(x))
        for x in cmd
    )


def stream_process(
    cmd,
    cwd,
    env,
    log_path,
):
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    interrupted = False

    try:
        assert (
            process.stdout
            is not None
        )

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log:
            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(line)
                log.flush()

        code = process.wait()

        return (
            code,
            interrupted,
        )

    except KeyboardInterrupt:
        interrupted = True

        print(
            "\n[SCHEDULER] Ctrl+C: "
            "stopping current run...",
            flush=True,
        )

        try:
            os.killpg(
                process.pid,
                signal.SIGTERM,
            )
        except ProcessLookupError:
            pass

        try:
            process.wait(
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(
                "[SCHEDULER] SIGTERM timeout; "
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

            process.wait()

        return (
            process.returncode,
            interrupted,
        )


def make_result_marker(
    run_dir,
    row,
    status,
    exit_code,
):
    marker = {
        "run_id":
            row["run_id"],

        "job_type":
            row["job_type"],

        "backbone":
            row["backbone"],

        "family":
            row["family"],

        "method":
            row["method"],

        "seed":
            row["seed"],

        "status":
            status,

        "exit_code":
            exit_code,

        "updated_at":
            now(),
    }

    write_json(
        run_dir
        / "scheduler_status.json",
        marker,
    )


def print_plan_summary(rows):
    counts = {}

    for row in rows:
        key = (
            row.get(
                "status",
                "planned",
            )
            or "planned"
        )

        counts[key] = (
            counts.get(
                key,
                0,
            )
            + 1
        )

    print()
    print("=" * 72)
    print(
        "Experiment plan status"
    )
    print("=" * 72)

    print(
        f"Total rows : {len(rows)}"
    )

    for key in sorted(counts):
        print(
            f"{key:<12}: "
            f"{counts[key]}"
        )

    print("=" * 72)


def main():
    args = parse_args()

    args.plan = (
        args.plan.expanduser().resolve()
    )

    args.run_root = (
        args.run_root.expanduser().resolve()
    )

    args.research_manifest = (
        args.research_manifest
        .expanduser()
        .resolve()
    )

    args.train_entry = (
        args.train_entry
        .expanduser()
        .resolve()
    )

    args.eval_entry = (
        args.eval_entry
        .expanduser()
        .resolve()
    )

    rows, fields = load_plan(
        args.plan
    )

    by_id = row_index(rows)

    # Convert stale "running" states
    # into interrupted before scheduling.
    changed = False

    for row in rows:
        if row["status"] == "running":
            row["status"] = "interrupted"
            row["last_error"] = (
                "stale running state "
                "recovered by scheduler"
            )
            changed = True

    if changed:
        atomic_write_plan(
            args.plan,
            rows,
            fields,
        )

    print_plan_summary(rows)

    candidates = [
        row
        for row in rows
        if selected(
            row,
            args,
        )
        and should_run(
            row,
            args,
        )
    ]

    if args.limit > 0:
        candidates = candidates[
            :args.limit
        ]

    print(
        f"\nSelected jobs: "
        f"{len(candidates)}"
    )

    if not candidates:
        print(
            "Nothing to run."
        )
        return 0

    args.run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_this_session = 0

    for number, row in enumerate(
        candidates,
        start=1,
    ):
        run_id = row["run_id"]

        ready, reason = (
            dependency_ready(
                row,
                by_id,
                args,
            )
        )

        if not ready:
            print()
            print(
                f"[SKIP] {run_id}"
            )
            print(
                f"       {reason}"
            )
            continue

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

        if row["job_type"] == "train":
            cmd = train_command(
                row,
                args,
                run_dir,
            )

        elif row["job_type"] == "eval":
            cmd = eval_command(
                row,
                args,
                run_dir,
                by_id,
            )

        else:
            print(
                f"[SKIP] unknown job_type: "
                f"{row['job_type']}"
            )
            continue

        print()
        print("#" * 72)
        print(
            f"[{number}/{len(candidates)}] "
            f"{run_id}"
        )
        print(
            f"type      : "
            f"{row['job_type']}"
        )
        print(
            f"backbone  : "
            f"{row['backbone']}"
        )
        print(
            f"method    : "
            f"{row['method']}"
        )
        print(
            f"seed      : "
            f"{row['seed']}"
        )
        print(
            f"output    : "
            f"{run_dir}"
        )
        print(
            "command   : "
            + print_command(cmd)
        )
        print("#" * 72)

        if args.dry_run:
            continue

        entry = (
            args.train_entry
            if row["job_type"] == "train"
            else args.eval_entry
        )

        if not entry.exists():
            raise FileNotFoundError(
                f"Entrypoint missing: {entry}"
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

        atomic_write_plan(
            args.plan,
            rows,
            fields,
        )

        make_result_marker(
            run_dir,
            row,
            "running",
            None,
        )

        env = os.environ.copy()

        env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(
            args.gpu
        )

        env[
            "PYTHONUNBUFFERED"
        ] = "1"

        try:
            code, interrupted = (
                stream_process(
                    cmd=cmd,
                    cwd=ROOT,
                    env=env,
                    log_path=log_path,
                )
            )

        except Exception as exc:
            code = -1
            interrupted = False

            row["last_error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            traceback.print_exc()

        row["finished_at"] = now()
        row["exit_code"] = str(code)

        if interrupted:
            row["status"] = (
                "interrupted"
            )

            row["last_error"] = (
                "interrupted by user"
            )

            row["result_path"] = str(
                run_dir
            )

            make_result_marker(
                run_dir,
                row,
                "interrupted",
                code,
            )

            atomic_write_plan(
                args.plan,
                rows,
                fields,
            )

            print(
                "\n[SCHEDULER] "
                "Stopped by user."
            )

            print(
                "[SCHEDULER] "
                "Resume with:"
            )

            print(
                "  --retry-interrupted"
            )

            return 130

        if code == 0:
            row["status"] = (
                "completed"
            )

            row["result_path"] = str(
                run_dir
            )

            row["last_error"] = ""

            completed_this_session += 1

            make_result_marker(
                run_dir,
                row,
                "completed",
                code,
            )

            print(
                f"[DONE] {run_id}"
            )

        else:
            row["status"] = "failed"

            row["result_path"] = str(
                run_dir
            )

            if not row["last_error"]:
                row["last_error"] = (
                    f"process exited "
                    f"with code {code}"
                )

            make_result_marker(
                run_dir,
                row,
                "failed",
                code,
            )

            print(
                f"[FAILED] {run_id} "
                f"exit_code={code}"
            )

        atomic_write_plan(
            args.plan,
            rows,
            fields,
        )

        # Refresh dependency lookup.
        by_id = row_index(rows)

    print()
    print("=" * 72)
    print(
        f"Completed this session: "
        f"{completed_this_session}"
    )
    print("=" * 72)

    print_plan_summary(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
