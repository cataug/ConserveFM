from pathlib import Path
import os
import pandas as pd

ROOT = Path.home() / "ConserveFM"

PLAN = ROOT / "manifests/experiments_v1/planned_runs.csv"

OUT = ROOT / "reports/prelim20_salvage"
CKROOT = ROOT / "runs/prelim20_ckpts"
TRAINROOT = ROOT / "runs/prelim20_train"

OUT.mkdir(parents=True, exist_ok=True)
CKROOT.mkdir(parents=True, exist_ok=True)
TRAINROOT.mkdir(parents=True, exist_ok=True)

p = pd.read_csv(
    PLAN,
    keep_default_na=False,
)

train = p[
    p["job_type"].astype(str).str.lower() == "train"
].copy()

rows = []

for _, r in train.iterrows():

    run_id = str(r["run_id"])
    backbone = str(r["backbone"])
    family = str(r.get("family", "main"))
    method = str(r["method"])
    seed = int(r["seed"])

    old = ROOT / "runs" / run_id

    best = old / "checkpoint_best.pt"
    last = old / "checkpoint_last.pt"
    hist = old / "train_history.csv"

    source = None

    if best.exists():
        source = best
    elif last.exists():
        source = last

    prior_step = 0
    prior_best_val = float("nan")
    prior_epochs = 0

    if hist.exists():
        try:
            h = pd.read_csv(hist)

            if "global_step" in h:
                prior_step = int(
                    pd.to_numeric(
                        h["global_step"],
                        errors="coerce",
                    ).max()
                )

            if "val_repair_mse" in h:
                prior_best_val = float(
                    pd.to_numeric(
                        h["val_repair_mse"],
                        errors="coerce",
                    ).min()
                )

            prior_epochs = len(h)

        except Exception:
            pass

    if source is not None:

        # Isolated screening checkpoint directory.
        # No mutation of original runs.
        ckdir = CKROOT / run_id
        ckdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        link = ckdir / "checkpoint_best.pt"

        if link.exists() or link.is_symlink():
            link.unlink()

        os.symlink(
            source.resolve(),
            link,
        )

        needs_train = 0
        source_kind = (
            "reuse_best"
            if source.name == "checkpoint_best.pt"
            else "reuse_last"
        )

    else:

        ckdir = TRAINROOT / run_id
        ckdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        needs_train = 1
        source_kind = "budget_train"

    rows.append({
        "run_id": run_id,
        "backbone": backbone,
        "family": family,
        "method": method,
        "seed": seed,

        "checkpoint_dir":
            str(ckdir),

        "needs_train":
            needs_train,

        "source_kind":
            source_kind,

        "source_checkpoint":
            str(source) if source else "",

        "prior_global_step":
            prior_step,

        "prior_best_val":
            prior_best_val,

        "prior_history_rows":
            prior_epochs,
    })

out = pd.DataFrame(rows)

out.to_csv(
    OUT / "jobs.csv",
    index=False,
)

out.to_csv(
    OUT / "jobs.tsv",
    index=False,
    sep="\t",
)

print("=" * 100)
print("PRELIMINARY 20-MIN SALVAGE")
print("=" * 100)

print("train jobs       :", len(out))
print(
    "checkpoint reuse:",
    int((out.needs_train == 0).sum())
)
print(
    "need budget train:",
    int((out.needs_train == 1).sum())
)

print("\nBY BACKBONE")
print(
    out.groupby(
        ["backbone", "needs_train"]
    )
    .size()
    .to_string()
)

print("\nREUSED COMPUTE")
print(
    out[
        out.needs_train == 0
    ][
        [
            "run_id",
            "prior_global_step",
            "prior_best_val",
            "source_kind",
        ]
    ]
    .sort_values(
        "prior_global_step",
        ascending=False,
    )
    .head(20)
    .to_string(
        index=False
    )
)

print(
    "\nSaved:",
    OUT / "jobs.tsv"
)
