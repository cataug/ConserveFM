from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "ConserveFM"
SRC = ROOT / "reports/final_metrics_s42"
OUT = SRC / "paper"
OUT.mkdir(parents=True, exist_ok=True)

final = pd.read_csv(SRC / "FINAL_SUMMARY.csv")
stress = pd.read_csv(SRC / "stress_conditions.csv")
stress_macro = pd.read_csv(SRC / "stress_macro_summary.csv")
stress_corr = pd.read_csv(SRC / "stress_by_corruption.csv")
stress_sev = pd.read_csv(SRC / "stress_by_severity.csv")
stress_lead = pd.read_csv(SRC / "stress_by_lead.csv")

MODEL_ORDER = [
    "No_Pretrain",
    "Direct_State",
    "Smart_Final",
    "OLD_Full",
]

DISPLAY = {
    "No_Pretrain": "No corruption pretraining",
    "Direct_State": "Direct state prediction",
    "Smart_Final": "ConserveFM (smart-final)",
    "OLD_Full": "ConserveFM (original)",
}

def ordered(df):
    x = df.copy()
    x["_ord"] = x["model"].map(
        {m:i for i,m in enumerate(MODEL_ORDER)}
    ).fillna(999)
    return x.sort_values("_ord").drop(columns="_ord")

def fmt(x, n=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{n}f}"

def pct_delta(a, b):
    # reduction of a relative to b; positive = a better if lower-is-better
    return 100.0 * (b - a) / b

def save_table(df, stem, caption=None, label=None):
    df.to_csv(OUT / f"{stem}.csv", index=False)

    # Write Markdown without optional pandas/tabulate dependency.
    cols = [str(c) for c in df.columns]

    def md_cell(v):
        if pd.isna(v):
            return "--"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v).replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(md_cell(row[c]) for c in df.columns)
            + " |"
        )

    (OUT / f"{stem}.md").write_text(
        "\n".join(lines) + "\n"
    )

    latex = df.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label,
    )
    (OUT / f"{stem}.tex").write_text(latex)

# ============================================================
# TABLE 1: MAIN RESULTS
# ============================================================

t1 = ordered(final)

t1 = pd.DataFrame({
    "Method": t1["model"].map(DISPLAY),
    "Clean NRMSE ↓": t1["clean_nrmse"],
    "Clean ACC ↑": t1["clean_acc"],
    "Stress NRMSE ↓": t1["macro_stress_nrmse"],
    "Stress ACC ↑": t1["macro_stress_acc"],
    "Loc. IoU ↑": t1["macro_localization_iou"],
    "Type Acc. ↑": t1["macro_type_accuracy"],
    "|Δx| ↓": t1["macro_repair_l1_norm"],
})

save_table(
    t1,
    "table1_main_results",
    caption="Main clean and stress-test results on ClimaX, seed 42.",
    label="tab:main_results",
)

# ============================================================
# TABLE 2: STRESS BREAKDOWN
# NRMSE by corruption + severity + lead
# ============================================================

def pivot_metric(df, index_col, prefix):
    p = df.pivot(
        index="model",
        columns=index_col,
        values="repaired_nrmse",
    )
    p.columns = [f"{prefix}{c}" for c in p.columns]
    return p

pc = pivot_metric(
    stress_corr,
    "corruption",
    "",
)

ps = pivot_metric(
    stress_sev,
    "severity",
    "sev=",
)

pl = pivot_metric(
    stress_lead,
    "lead_hours",
    "lead=",
)

t2 = (
    final[["model", "macro_stress_nrmse"]]
    .set_index("model")
    .join(pc)
    .join(ps)
    .join(pl)
    .reset_index()
)

t2 = ordered(t2)

rename = {
    "model": "Method",
    "macro_stress_nrmse": "Macro",
    "moisture": "Moisture",
    "hydrostatic": "Hydrostatic",
    "advection": "Advection",
    "spectral": "Spectral",
    "range_extreme": "Range/extreme",
    "sev=0.25": "Sev .25",
    "sev=0.5": "Sev .5",
    "sev=1.0": "Sev 1",
    "sev=2.0": "Sev 2",
    "lead=6": "6 h",
    "lead=24": "24 h",
    "lead=72": "72 h",
}

t2 = t2.rename(columns=rename)
t2["Method"] = t2["Method"].map(DISPLAY)

save_table(
    t2,
    "table2_stress_breakdown",
    caption="Normalized RMSE under synthetic physical stress, broken down by corruption type, severity, and forecast lead.",
    label="tab:stress_breakdown",
)

# ============================================================
# TABLE 3: PHYSICS + DIAGNOSTICS
#
# ratio = repaired physics violation / corrupted-input violation
# <1 improves physics violation
# >1 makes it worse
# ============================================================

constraints = [
    "moisture",
    "hydrostatic",
    "advection",
    "spectral",
]

rows = []

for _, r in stress_macro.iterrows():
    row = {
        "model": r["model"],
    }

    ratios = []

    for c in constraints:
        before = r.get(f"corrupted_input_phys_{c}", np.nan)
        after = r.get(f"repaired_phys_{c}", np.nan)

        ratio = (
            after / before
            if pd.notna(before)
            and pd.notna(after)
            and abs(before) > 1e-20
            else np.nan
        )

        row[c] = ratio

        if pd.notna(ratio) and ratio > 0:
            ratios.append(ratio)

    # Geometric mean avoids one differently-scaled constraint
    # dominating the summary.
    row["physics_ratio_gm"] = (
        float(np.exp(np.mean(np.log(ratios))))
        if ratios else np.nan
    )

    f = final[final.model == r["model"]].iloc[0]

    row["loc"] = f["macro_localization_iou"]
    row["type"] = f["macro_type_accuracy"]
    row["repair"] = f["macro_repair_l1_norm"]
    row["stress"] = f["macro_stress_nrmse"]

    rows.append(row)

t3 = ordered(pd.DataFrame(rows))

t3 = t3.rename(columns={
    "model": "Method",
    "moisture": "Moisture ratio ↓",
    "hydrostatic": "Hydrostatic ratio ↓",
    "advection": "Advection ratio ↓",
    "spectral": "Spectral ratio ↓",
    "physics_ratio_gm": "Physics ratio GM ↓",
    "loc": "Loc. IoU ↑",
    "type": "Type Acc. ↑",
    "repair": "|Δx| ↓",
    "stress": "Stress NRMSE ↓",
})

t3["Method"] = t3["Method"].map(DISPLAY)

save_table(
    t3,
    "table3_physics_diagnostics",
    caption=(
        "Physical-violation repair ratios and diagnostic performance. "
        "A physics ratio below one indicates that repair reduced the "
        "corresponding violation relative to the corrupted input."
    ),
    label="tab:physics_diagnostics",
)

# ============================================================
# EXTRA: per-condition winners
# ============================================================

w = stress[
    [
        "model",
        "lead_hours",
        "corruption",
        "severity",
        "repaired_nrmse",
    ]
].copy()

idx = (
    w.groupby(
        ["lead_hours", "corruption", "severity"]
    )["repaired_nrmse"]
    .idxmin()
)

wins = (
    w.loc[idx]
    .groupby("model")
    .size()
    .reindex(MODEL_ORDER, fill_value=0)
    .reset_index(name="stress_condition_wins")
)

wins["Method"] = wins["model"].map(DISPLAY)
wins = wins[
    ["Method", "stress_condition_wins"]
]

wins.to_csv(
    OUT / "stress_condition_wins.csv",
    index=False,
)

# ============================================================
# DECISION
# ============================================================

f = final.set_index("model")

predictive_winner = f["macro_stress_nrmse"].idxmin()
clean_winner = f["clean_nrmse"].idxmin()

phys = pd.DataFrame(rows).set_index("model")
physics_winner = (
    phys["physics_ratio_gm"].idxmin()
    if phys["physics_ratio_gm"].notna().any()
    else None
)

no_pre = f.loc["No_Pretrain"]
direct = f.loc["Direct_State"]
smart = f.loc["Smart_Final"]
old = f.loc["OLD_Full"]

lines = []

lines.append("FINAL EXPERIMENTAL DECISION — CLIMAX SEED 42")
lines.append("=" * 72)
lines.append("")

lines.append(
    f"Best clean NRMSE: {clean_winner} "
    f"({f.loc[clean_winner, 'clean_nrmse']:.5f})"
)

lines.append(
    f"Best macro stress NRMSE: {predictive_winner} "
    f"({f.loc[predictive_winner, 'macro_stress_nrmse']:.5f})"
)

if physics_winner is not None:
    lines.append(
        f"Best aggregate physics repair ratio: {physics_winner} "
        f"({phys.loc[physics_winner, 'physics_ratio_gm']:.5f})"
    )

lines.append("")

lines.append(
    "No_Pretrain vs OLD_Full stress NRMSE: "
    f"{no_pre['macro_stress_nrmse']:.5f} vs "
    f"{old['macro_stress_nrmse']:.5f} "
    f"({pct_delta(no_pre['macro_stress_nrmse'], old['macro_stress_nrmse']):+.2f}% reduction)"
)

lines.append(
    "Smart_Final vs OLD_Full stress NRMSE: "
    f"{smart['macro_stress_nrmse']:.5f} vs "
    f"{old['macro_stress_nrmse']:.5f} "
    f"({pct_delta(smart['macro_stress_nrmse'], old['macro_stress_nrmse']):+.2f}% reduction)"
)

lines.append(
    "Smart_Final vs No_Pretrain stress NRMSE: "
    f"{smart['macro_stress_nrmse']:.5f} vs "
    f"{no_pre['macro_stress_nrmse']:.5f} "
    f"({pct_delta(smart['macro_stress_nrmse'], no_pre['macro_stress_nrmse']):+.2f}% reduction; "
    "negative means Smart_Final is worse)"
)

lines.append("")

random_type = 1.0 / 5.0

lines.append(
    f"Smart_Final type accuracy = "
    f"{smart['macro_type_accuracy']:.4f}; "
    f"5-class random baseline = {random_type:.4f}."
)

lines.append(
    f"Smart_Final localization IoU = "
    f"{smart['macro_localization_iou']:.4f}."
)

lines.append("")

# Conservative automatic conclusion.
if predictive_winner == "No_Pretrain":
    lines.append(
        "RECOMMENDATION: use No_Pretrain as the primary predictive "
        "ConserveFM variant for the current ClimaX results."
    )

    lines.append(
        "Corruption pretraining / adaptive diagnostic machinery should "
        "be reported as an ablation unless Table 3 shows a sufficiently "
        "large physics-consistency advantage to justify its NRMSE cost."
    )

elif predictive_winner == "Smart_Final":
    lines.append(
        "RECOMMENDATION: use Smart_Final as the primary ConserveFM variant."
    )

else:
    lines.append(
        f"RECOMMENDATION: strongest predictive variant is "
        f"{predictive_winner}; do not label another variant as primary "
        "without an explicit physics/interpretability trade-off."
    )

lines.append("")
lines.append(
    "Do not use mixed-unit physical RMSE as the headline metric. "
    "Use clean/stress NRMSE and ACC, with per-variable physical RMSE "
    "and constraint-specific physics residuals as supporting metrics."
)

decision = "\n".join(lines)

(OUT / "FINAL_DECISION.txt").write_text(
    decision + "\n"
)

# ============================================================
# PRINT
# ============================================================

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 220)

print("\n" + "=" * 110)
print("TABLE 1 — MAIN")
print("=" * 110)
print(
    t1.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 110)
print("TABLE 2 — STRESS BREAKDOWN")
print("=" * 110)
print(
    t2.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 110)
print("TABLE 3 — PHYSICS / DIAGNOSTICS")
print("=" * 110)
print(
    t3.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + decision)

print("\nOUTPUT:")
print(OUT)
