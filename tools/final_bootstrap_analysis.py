from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "ConserveFM"
SRC = ROOT / "reports/final_metrics_s42"
OUT = SRC / "paper"
OUT.mkdir(parents=True, exist_ok=True)

stress = pd.read_csv(SRC / "stress_conditions.csv")
final = pd.read_csv(SRC / "FINAL_SUMMARY.csv")

MODELS = [
    "No_Pretrain",
    "Direct_State",
    "Smart_Final",
    "OLD_Full",
]

DISPLAY = {
    "No_Pretrain": "ConserveFM-Acc",
    "Direct_State": "Direct state",
    "Smart_Final": "ConserveFM-Phys",
    "OLD_Full": "Original full",
}

KEYS = ["lead_hours", "corruption", "severity"]

# ------------------------------------------------------------------
# Verify exactly the same 60 conditions exist for every model.
# ------------------------------------------------------------------
for m in MODELS:
    q = stress[stress.model == m]
    print(m, "conditions =", len(q))

base_keys = (
    stress[stress.model == MODELS[0]][KEYS]
    .sort_values(KEYS)
    .reset_index(drop=True)
)

for m in MODELS[1:]:
    k = (
        stress[stress.model == m][KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    if not base_keys.equals(k):
        raise SystemExit(f"Condition mismatch for {m}")

print("Paired-condition check: OK")

# ------------------------------------------------------------------
# Wide paired table.
# ------------------------------------------------------------------
wide = None

keep_metrics = [
    "repaired_nrmse",
    "repaired_acc",
    "localization_iou",
    "type_accuracy",
    "repair_l1_norm",
]

for m in MODELS:
    q = (
        stress[stress.model == m]
        [KEYS + keep_metrics]
        .sort_values(KEYS)
        .reset_index(drop=True)
        .copy()
    )

    q = q.rename(columns={
        c: f"{m}__{c}"
        for c in keep_metrics
    })

    if wide is None:
        wide = q
    else:
        wide = wide.merge(
            q,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

N = len(wide)
assert N == 60, N

# ------------------------------------------------------------------
# Bootstrap helpers.
# Paired resampling: resample the SAME condition indices for all models.
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
B = 20000

boot_idx = rng.integers(
    0,
    N,
    size=(B, N),
)

def ci_mean(values):
    values = np.asarray(values, dtype=float)

    boot = values[boot_idx].mean(axis=1)

    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )

def ci_paired_diff(a, b):
    """
    a - b.
    Negative is better when metric is lower-is-better.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    d = a - b
    boot = d[boot_idx].mean(axis=1)

    return (
        float(d.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        float((boot < 0).mean()),
    )

def fmt_ci(mu, lo, hi):
    return f"{mu:.5f} [{lo:.5f}, {hi:.5f}]"

# ------------------------------------------------------------------
# Table A: absolute stress metrics + bootstrap CI.
# ------------------------------------------------------------------
rows = []

for m in MODELS:
    nrmse = wide[f"{m}__repaired_nrmse"].to_numpy()
    acc = wide[f"{m}__repaired_acc"].to_numpy()

    a = ci_mean(nrmse)
    b = ci_mean(acc)

    f = final[final.model == m].iloc[0]

    rows.append({
        "Method": DISPLAY[m],
        "Clean NRMSE": f["clean_nrmse"],
        "Clean ACC": f["clean_acc"],
        "Stress NRMSE mean": a[0],
        "Stress NRMSE CI low": a[1],
        "Stress NRMSE CI high": a[2],
        "Stress ACC mean": b[0],
        "Stress ACC CI low": b[1],
        "Stress ACC CI high": b[2],
        "Loc IoU": f["macro_localization_iou"],
        "Type Acc": f["macro_type_accuracy"],
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "final_results_bootstrap95.csv",
    index=False,
)

print("\n" + "=" * 105)
print("FINAL RESULTS — PAIRED-CONDITION BOOTSTRAP 95% CI")
print("=" * 105)

for _, r in summary.iterrows():
    print(
        f"{r['Method']:18s} "
        f"clean={r['Clean NRMSE']:.5f}  "
        f"stress="
        f"{r['Stress NRMSE mean']:.5f} "
        f"[{r['Stress NRMSE CI low']:.5f}, "
        f"{r['Stress NRMSE CI high']:.5f}]  "
        f"ACC={r['Stress ACC mean']:.5f}"
    )

# ------------------------------------------------------------------
# Table B: paired comparisons against No_Pretrain / ConserveFM-Acc.
# ------------------------------------------------------------------
ref = "No_Pretrain"
refv = wide[f"{ref}__repaired_nrmse"].to_numpy()

comparisons = []

for m in MODELS:
    if m == ref:
        continue

    v = wide[f"{m}__repaired_nrmse"].to_numpy()

    # ref - competitor:
    # negative => reference lower/better.
    mu, lo, hi, pneg = ci_paired_diff(
        refv,
        v,
    )

    comparisons.append({
        "Comparison":
            f"{DISPLAY[ref]} - {DISPLAY[m]}",
        "Mean NRMSE difference": mu,
        "95% CI low": lo,
        "95% CI high": hi,
        "Bootstrap P(diff < 0)": pneg,
    })

cmp = pd.DataFrame(comparisons)

cmp.to_csv(
    OUT / "paired_nrmse_comparisons.csv",
    index=False,
)

print("\n" + "=" * 105)
print("PAIRED STRESS-NRMSE DIFFERENCES")
print("negative => ConserveFM-Acc is better")
print("=" * 105)
print(
    cmp.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# ------------------------------------------------------------------
# Physics ratio per CONDITION + bootstrap geometric mean.
# ------------------------------------------------------------------
constraints = [
    "moisture",
    "hydrostatic",
    "advection",
    "spectral",
]

physics_rows = []

for m in MODELS:
    q = (
        stress[stress.model == m]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    per_constraint = {}
    log_all = []

    for c in constraints:
        before = q[
            f"corrupted_input_phys_{c}"
        ].to_numpy(dtype=float)

        after = q[
            f"repaired_phys_{c}"
        ].to_numpy(dtype=float)

        ratio = after / np.maximum(
            np.abs(before),
            1e-20,
        )

        ratio = np.maximum(ratio, 1e-20)

        logs = np.log(ratio)
        log_all.append(logs)

        # Bootstrap geometric mean over the same 60 conditions.
        boots = np.exp(
            logs[boot_idx].mean(axis=1)
        )

        per_constraint[c] = (
            float(np.exp(logs.mean())),
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)),
        )

    # Each condition contributes all four constraints equally in log-space.
    L = np.stack(log_all, axis=1)  # [60, 4]

    condition_log_gm = L.mean(axis=1)

    boots = np.exp(
        condition_log_gm[boot_idx].mean(axis=1)
    )

    gm = (
        float(np.exp(condition_log_gm.mean())),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )

    row = {
        "Method": DISPLAY[m],
        "Physics GM": gm[0],
        "Physics GM CI low": gm[1],
        "Physics GM CI high": gm[2],
    }

    for c in constraints:
        mu, lo, hi = per_constraint[c]
        row[f"{c} ratio"] = mu
        row[f"{c} CI low"] = lo
        row[f"{c} CI high"] = hi

    physics_rows.append(row)

phys = pd.DataFrame(physics_rows)

phys.to_csv(
    OUT / "physics_ratios_bootstrap95.csv",
    index=False,
)

print("\n" + "=" * 105)
print("PHYSICS VIOLATION RATIOS — BOOTSTRAP 95% CI")
print("<1 = violation reduced")
print("=" * 105)

for _, r in phys.iterrows():
    print(
        f"{r['Method']:18s} "
        f"GM={r['Physics GM']:.4f} "
        f"[{r['Physics GM CI low']:.4f}, "
        f"{r['Physics GM CI high']:.4f}] "
        f"| moist={r['moisture ratio']:.3f} "
        f"hydro={r['hydrostatic ratio']:.3f} "
        f"adv={r['advection ratio']:.3f} "
        f"spec={r['spectral ratio']:.3f}"
    )

# ------------------------------------------------------------------
# Pareto analysis: stress NRMSE vs physics GM.
# ------------------------------------------------------------------
merged = summary[
    ["Method", "Stress NRMSE mean"]
].merge(
    phys[
        ["Method", "Physics GM"]
    ],
    on="Method",
)

def dominated(i, df):
    a = df.iloc[i]

    for j in range(len(df)):
        if i == j:
            continue

        b = df.iloc[j]

        if (
            b["Stress NRMSE mean"]
            <= a["Stress NRMSE mean"]
            and
            b["Physics GM"]
            <= a["Physics GM"]
            and
            (
                b["Stress NRMSE mean"]
                < a["Stress NRMSE mean"]
                or
                b["Physics GM"]
                < a["Physics GM"]
            )
        ):
            return True

    return False

merged["Pareto"] = [
    not dominated(i, merged)
    for i in range(len(merged))
]

merged.to_csv(
    OUT / "accuracy_physics_pareto.csv",
    index=False,
)

print("\n" + "=" * 105)
print("ACCURACY–PHYSICS PARETO FRONT")
print("=" * 105)
print(
    merged.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print("\nSaved:")
print(OUT)
