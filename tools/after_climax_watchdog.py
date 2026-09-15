#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path.home() / "ConserveFM"

PY = (
    Path.home()
    / "Malashin_Projects/.venv_a100/bin/python"
)

DATA_PY = (
    ROOT
    / ".venv_data/bin/python"
)


def data_site():
    return subprocess.check_output(
        [
            str(DATA_PY),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        text=True,
    ).strip()


DS = data_site()

ENV = os.environ.copy()

ENV["CUDA_VISIBLE_DEVICES"] = "0"

ENV["PYTHONPATH"] = ":".join(
    [
        str(ROOT),
        str(
            ROOT
            / "external/fcnv2_runtime"
        ),
        str(
            ROOT
            / "external/ai-models-fourcastnetv2"
        ),
        str(
            ROOT
            / "external/Prithvi-WxC"
        ),
        DS,
        ENV.get(
            "PYTHONPATH",
            "",
        ),
    ]
)


def summary_ok(store):
    p = (
        ROOT
        / store
        / "generation_summary.json"
    )

    if not p.exists():
        return False

    try:
        x = json.loads(
            p.read_text()
        )
    except Exception:
        return False

    return (
        x.get("status")
        == "completed"
        and not any(
            int(v)
            for v in x.get(
                "missing",
                {},
            ).values()
        )
    )


def run(cmd):
    print()
    print(
        "[WATCHDOG] RUN:",
        " ".join(
            str(x)
            for x in cmd
        ),
        flush=True,
    )

    p = subprocess.Popen(
        [
            str(x)
            for x in cmd
        ],
        cwd=ROOT,
        env=ENV,
    )

    return p.wait()


def main():
    print(
        "[WATCHDOG] waiting for complete climax.zarr",
        flush=True,
    )

    while not summary_ok(
        "forecasts/climax.zarr"
    ):
        p = (
            ROOT
            / "forecasts/climax.zarr/"
            "generation_summary.json"
        )

        if p.exists():
            print(
                "[WATCHDOG] ClimaX summary exists "
                "but is not complete yet",
                flush=True,
            )
        else:
            print(
                "[WATCHDOG] ClimaX still running...",
                flush=True,
            )

        time.sleep(60)

    print(
        "[WATCHDOG] ClimaX COMPLETE",
        flush=True,
    )

    #
    # FourCastNet smoke.
    #
    smoke = (
        ROOT
        / "forecasts/fourcastnet_smoke.zarr"
    )

    if smoke.exists():
        shutil.rmtree(smoke)

    code = run(
        [
            PY,
            "-X",
            "faulthandler",
            "-u",
            ROOT
            / "tools/generate_fourcastnet_forecasts.py",

            "--output",
            smoke,

            "--max-contexts",
            "2",

            "--log-every",
            "1",

            "--overwrite",
        ]
    )

    if code != 0:
        print(
            f"[WATCHDOG] FourCastNet smoke FAILED code={code}",
            flush=True,
        )
        return code

    print(
        "[WATCHDOG] FourCastNet smoke OK",
        flush=True,
    )

    #
    # Full FCNv2, resume-safe.
    #
    code = run(
        [
            PY,
            "-X",
            "faulthandler",
            "-u",
            ROOT
            / "tools/generate_fourcastnet_forecasts.py",

            "--output",
            ROOT
            / "forecasts/fourcastnet.zarr",

            "--log-every",
            "25",
        ]
    )

    if code != 0:
        print(
            f"[WATCHDOG] FourCastNet full FAILED code={code}",
            flush=True,
        )
        return code

    if not summary_ok(
        "forecasts/fourcastnet.zarr"
    ):
        print(
            "[WATCHDOG] FourCastNet summary incomplete",
            flush=True,
        )
        return 3

    print(
        "[WATCHDOG] FourCastNet COMPLETE",
        flush=True,
    )

    #
    # Prithvi smoke.
    #
    smoke = (
        ROOT
        / "forecasts/prithvi_wxc_smoke.zarr"
    )

    if smoke.exists():
        shutil.rmtree(smoke)

    code = run(
        [
            PY,
            "-X",
            "faulthandler",
            "-u",
            ROOT
            / "tools/generate_prithvi_forecasts.py",

            "--output",
            smoke,

            "--max-contexts",
            "1",

            "--log-every",
            "1",

            "--overwrite",
        ]
    )

    if code != 0:
        print(
            f"[WATCHDOG] Prithvi smoke FAILED code={code}",
            flush=True,
        )
        return code

    print(
        "[WATCHDOG] Prithvi smoke OK",
        flush=True,
    )

    #
    # Full Prithvi, resume-safe.
    #
    code = run(
        [
            PY,
            "-X",
            "faulthandler",
            "-u",
            ROOT
            / "tools/generate_prithvi_forecasts.py",

            "--output",
            ROOT
            / "forecasts/prithvi_wxc.zarr",

            "--log-every",
            "10",
        ]
    )

    if code != 0:
        print(
            f"[WATCHDOG] Prithvi full FAILED code={code}",
            flush=True,
        )
        return code

    if not summary_ok(
        "forecasts/prithvi_wxc.zarr"
    ):
        print(
            "[WATCHDOG] Prithvi summary incomplete",
            flush=True,
        )
        return 4

    print()
    print(
        "[WATCHDOG] ALL THREE FORECAST STORES COMPLETE",
        flush=True,
    )

    #
    # Finally 138 jobs.
    #
    code = run(
        [
            PY,
            "-u",
            ROOT
            / "tools/run_parallel_plan.py",

            "--plan",
            ROOT
            / "manifests/experiments_v1/planned_runs.csv",

            "--run-root",
            ROOT
            / "runs",

            "--python",
            PY,

            "--data-site",
            DS,

            "--gpu",
            "0",

            "--max-parallel",
            "4",

            "--reserve-per-job-gib",
            "7",

            "--headroom-gib",
            "4",

            "--launch-gap",
            "10",

            "--poll-seconds",
            "10",
        ]
    )

    print(
        f"[WATCHDOG] 138-job scheduler finished code={code}",
        flush=True,
    )

    return code


if __name__ == "__main__":
    raise SystemExit(main())
