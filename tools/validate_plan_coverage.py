#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path.home() / "ConserveFM"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conservefm.channels import build_layout
from conservefm.model import ConserveRepair
from conservefm.registry import SUPPORTED_METHODS, resolve_method, validate_registry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "manifests/experiments_v1/planned_runs.csv",
    )
    p.add_argument("--synthetic-forward", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    validate_registry()
    with args.plan.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train = [r for r in rows if r["job_type"] == "train"]
    ev = [r for r in rows if r["job_type"] == "eval"]
    problems = []

    for r in train:
        if r["method"] not in SUPPORTED_METHODS:
            problems.append(f"TRAIN unsupported method: {r['run_id']} -> {r['method']}")
        else:
            resolve_method(r["method"])

    for r in ev:
        if r["method"] not in SUPPORTED_METHODS + ["raw_fm", "hard_projection"]:
            problems.append(f"EVAL unsupported method: {r['run_id']} -> {r['method']}")

    by_id = {r["run_id"]: r for r in rows}
    for r in ev:
        dep = (r.get("depends_on") or "").strip()
        if dep and dep not in by_id:
            problems.append(f"Missing dependency {dep} for {r['run_id']}")

    if args.synthetic_forward:
        layout = build_layout([50,100,150,200,250,300,400,500,600,700,850,925,1000])
        raw = torch.randn(1, layout.n_channels, 16, 12)
        prev = torch.randn_like(raw)
        lead = torch.tensor([24.0])
        for method in SUPPORTED_METHODS:
            cfg = resolve_method(method)
            model = ConserveRepair(layout.n_channels, cfg, width=16, depth=1)
            out = model(raw, prev, lead)
            assert out.state_norm.shape == raw.shape
            assert out.delta_norm.shape == raw.shape
            print(
                f"[FORWARD OK] {method:<30} "
                f"router={cfg.router_mode:<14} constraints={','.join(cfg.enabled_constraints) or '-'}"
            )

    print("=" * 72)
    print("ConserveFM plan coverage")
    print("=" * 72)
    print("rows              :", len(rows))
    print("train jobs         :", len(train))
    print("eval jobs          :", len(ev))
    print("train methods      :", len(set(r["method"] for r in train)))
    print("train backbones    :", sorted(set(r["backbone"] for r in train)))
    print("method counts      :", dict(Counter(r["method"] for r in train)))
    print("problems           :", len(problems))
    if problems:
        for p in problems:
            print("ERROR:", p)
        raise SystemExit(2)
    print("STATUS             : ALL PLAN METHODS RESOLVE")
    print("=" * 72)


if __name__ == "__main__":
    main()
