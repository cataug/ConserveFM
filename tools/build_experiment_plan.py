#!/usr/bin/env python3

import csv
import json
from pathlib import Path


ROOT = Path.home() / "ConserveFM"
OUT = ROOT / "manifests/experiments_v1"

SEEDS = [42, 43, 44]

PRIMARY = "climax"

TRANSFER = [
    "fourcastnet",
    "prithvi_wxc",
]


PRIMARY_TRAINABLE = [
    # Main
    ("main", "static_physics_loss"),
    ("main", "residual_repair"),
    ("main", "conservefm_full"),

    # Physics-corruption ablations
    ("corruption", "no_corruption_pretrain"),
    ("corruption", "no_moisture_corruption"),
    ("corruption", "no_hydrostatic_corruption"),
    ("corruption", "no_advection_corruption"),
    ("corruption", "no_spectral_corruption"),
    ("corruption", "no_range_extreme_corruption"),

    # Router ablations
    ("router", "equal_constraint_weights"),
    ("router", "static_learned_weights"),
    ("router", "manual_tuned_weights"),

    # Architecture ablations
    ("architecture", "no_localization_head"),
    ("architecture", "no_constraint_type_head"),
    ("architecture", "no_minimal_change"),
    ("architecture", "direct_state_prediction"),

    # Constraint ablations
    ("constraint", "no_moisture_constraint"),
    ("constraint", "no_hydrostatic_constraint"),
    ("constraint", "no_advection_constraint"),
    ("constraint", "no_spectral_constraint"),
]


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = []

    #
    # Primary backbone:
    # all main trainable methods + ablations
    #
    for family, method in PRIMARY_TRAINABLE:
        for seed in SEEDS:
            run_id = (
                f"train_{PRIMARY}_"
                f"{method}_s{seed}"
            )

            train.append({
                "run_id": run_id,
                "job_type": "train",
                "backbone": PRIMARY,
                "family": family,
                "method": method,
                "seed": seed,
                "depends_on": "",
                "status": "planned",
            })

    #
    # Backbone-transfer experiments:
    # Full ConserveFM only.
    #
    for backbone in TRANSFER:
        for seed in SEEDS:
            method = "conservefm_full"

            run_id = (
                f"train_{backbone}_"
                f"{method}_s{seed}"
            )

            train.append({
                "run_id": run_id,
                "job_type": "train",
                "backbone": backbone,
                "family": "backbone_transfer",
                "method": method,
                "seed": seed,
                "depends_on": "",
                "status": "planned",
            })

    assert len(train) == 66, len(train)

    eval_rows = []

    #
    # Every trained checkpoint gets one consolidated
    # evaluation job.
    #
    for row in train:
        eval_rows.append({
            "run_id":
                row["run_id"].replace(
                    "train_",
                    "eval_",
                    1,
                ),
            "job_type": "eval",
            "backbone": row["backbone"],
            "family": row["family"],
            "method": row["method"],
            "seed": row["seed"],
            "depends_on": row["run_id"],
            "eval_clean": True,
            "eval_stress": True,
            "eval_ghcn": True,
            "lead_hours": "6|24|72",
            "stress_types":
                "moisture|hydrostatic|"
                "advection|spectral|range_extreme",
            "severity":
                "0.25|0.5|1.0|2.0",
            "status": "planned",
        })

    #
    # Deterministic no-training baselines:
    # Raw + hard projection on all backbones.
    #
    for backbone in [
        PRIMARY,
        *TRANSFER,
    ]:
        for method in [
            "raw_fm",
            "hard_projection",
        ]:
            eval_rows.append({
                "run_id":
                    f"eval_{backbone}_{method}_det",
                "job_type": "eval",
                "backbone": backbone,
                "family": "baseline",
                "method": method,
                "seed": "det",
                "depends_on": "",
                "eval_clean": True,
                "eval_stress": True,
                "eval_ghcn": True,
                "lead_hours": "6|24|72",
                "stress_types":
                    "moisture|hydrostatic|"
                    "advection|spectral|range_extreme",
                "severity":
                    "0.25|0.5|1.0|2.0",
                "status": "planned",
            })

    assert len(eval_rows) == 72, len(eval_rows)

    write_csv(
        OUT / "train_runs.csv",
        train,
    )

    write_csv(
        OUT / "eval_runs.csv",
        eval_rows,
    )

    combined = []

    for row in train:
        combined.append({
            **row,
            "eval_clean": "",
            "eval_stress": "",
            "eval_ghcn": "",
            "lead_hours": "",
            "stress_types": "",
            "severity": "",
        })

    combined.extend(eval_rows)

    write_csv(
        OUT / "planned_runs.csv",
        combined,
    )

    summary = {
        "seeds": SEEDS,
        "primary_backbone": PRIMARY,
        "transfer_backbones": TRANSFER,

        "primary_trainable_configs":
            len(PRIMARY_TRAINABLE),

        "transfer_trainable_configs":
            len(TRANSFER),

        "unique_trainable_configs":
            22,

        "training_runs":
            len(train),

        "deterministic_eval_baselines":
            6,

        "evaluation_jobs":
            len(eval_rows),

        "total_scheduler_jobs":
            len(train) + len(eval_rows),

        "lead_hours": [
            6,
            24,
            72,
        ],

        "stress_types": [
            "moisture",
            "hydrostatic",
            "advection",
            "spectral",
            "range_extreme",
        ],

        "severity_levels": [
            0.25,
            0.5,
            1.0,
            2.0,
        ],
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 66)
    print("ConserveFM experiment plan")
    print("=" * 66)
    print(
        "Primary trainable configs :",
        len(PRIMARY_TRAINABLE),
    )
    print(
        "Transfer configs          :",
        len(TRANSFER),
    )
    print(
        "Unique trainable configs  :",
        22,
    )
    print(
        "Training runs             :",
        len(train),
    )
    print(
        "Evaluation jobs           :",
        len(eval_rows),
    )
    print(
        "TOTAL scheduler jobs      :",
        len(train) + len(eval_rows),
    )
    print("=" * 66)


if __name__ == "__main__":
    main()
