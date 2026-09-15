#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--run-id", required=True)
p.add_argument("--backbone", required=True)
p.add_argument("--family", required=True)
p.add_argument("--method", required=True)
p.add_argument("--seed", required=True)
p.add_argument("--research-manifest", required=True)
p.add_argument("--output-dir", required=True)
args = p.parse_args()

seconds = int(os.environ.get("SMOKE_SECONDS", "120"))

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)

print("=" * 70, flush=True)
print("ConserveFM SMOKE TRAIN", flush=True)
print("run_id   :", args.run_id, flush=True)
print("backbone :", args.backbone, flush=True)
print("method   :", args.method, flush=True)
print("seed     :", args.seed, flush=True)
print("duration :", seconds, "sec", flush=True)
print("=" * 70, flush=True)

started = time.time()

step = 0
while True:
    elapsed = time.time() - started

    if elapsed >= seconds:
        break

    step += 1

    fake_loss = 1.0 / (1.0 + step * 0.05)

    print(
        f"[SMOKE] "
        f"step={step:03d} | "
        f"elapsed={elapsed:6.1f}s | "
        f"loss={fake_loss:.5f}",
        flush=True,
    )

    time.sleep(5)

result = {
    "run_id": args.run_id,
    "status": "ok",
    "smoke": True,
    "duration_seconds": round(time.time() - started, 2),
    "backbone": args.backbone,
    "method": args.method,
    "seed": args.seed,
}

(out / "result.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

(out / "checkpoint_best.pt").write_text(
    "SMOKE CHECKPOINT ONLY\n",
    encoding="utf-8",
)

print("=" * 70, flush=True)
print("[SMOKE] SUCCESS", flush=True)
print("result:", out / "result.json", flush=True)
print("=" * 70, flush=True)
