"""Build a clean per-FAQ results table from the latest arm-comparison JSON.

Reads the most recent eval/results/arm_comparison_*.json and produces a
Markdown table grouped by query type (A/B/C/D), with per-type subtotals and
a grand total. Writes to eval/results/results_table.md and prints to stdout.

No re-running of the model - pure formatting of already-saved results.

Usage (from project root):
    uv run python src/evaluation/make_results_table.py

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Map the raw type strings to readable labels + the report's A/B/C/D grouping.
TYPE_LABELS = {
    "explicit_sustainability": ("A", "Explicit sustainability"),
    "implicit_sustainability": ("B", "Implicit / trending"),
    "how_to_sustainability": ("C", "How-to"),
    "accommodation_activities": ("D", "Accommodation & activities"),
}


def load_latest() -> dict:
    files = glob.glob(str(RESULTS_DIR / "arm_comparison_*.json"))
    if not files:
        raise FileNotFoundError("No arm_comparison_*.json in eval/results/")
    latest = max(files, key=os.path.getmtime)
    print(f"Reading: {latest}\n")
    with open(latest) as f:
        return json.load(f)


def main() -> None:
    data = load_latest()
    results = data["results"]

    lines = []
    lines.append("# Phase 6 Results: Arm 1 (vanilla) vs Arm 3 (full system)\n")
    lines.append(f"Generated from run: {data['generated']}  |  "
                 f"FAQs: {data['n_succeeded']}/{data['n_faqs']} succeeded\n")
    lines.append(
        "| FAQ | Type | Region | A1 codes | A3 codes (grounded) | "
        "A3 ungrounded (stripped) | A3 schemes |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    # Group by type in A/B/C/D order.
    order = ["explicit_sustainability", "implicit_sustainability",
             "how_to_sustainability", "accommodation_activities"]

    grand = {"a1": 0, "a3": 0, "stripped": 0, "attempted": 0}

    for t in order:
        group = [r for r in results if r["type"] == t]
        if not group:
            continue
        letter, label = TYPE_LABELS.get(t, ("?", t))
        sub = {"a1": 0, "a3": 0, "stripped": 0}
        for r in group:
            a1 = r["arm1_baseline"]
            a3 = r["arm3_full_system"]
            schemes = ", ".join(a3["schemes_mentioned"]) or "-"
            lines.append(
                f"| {r['id']} | {letter} | {r.get('region','-')} | "
                f"{a1['n_codes']} | {a3['n_codes']} | "
                f"{a3['n_stripped']} | {schemes} |"
            )
            sub["a1"] += a1["n_codes"]
            sub["a3"] += a3["n_codes"]
            sub["stripped"] += a3["n_stripped"]
        # Per-type subtotal row.
        lines.append(
            f"| **Type {letter} subtotal** | | | **{sub['a1']}** | "
            f"**{sub['a3']}** | **{sub['stripped']}** | |"
        )
        grand["a1"] += sub["a1"]
        grand["a3"] += sub["a3"]
        grand["stripped"] += sub["stripped"]

    grand["attempted"] = grand["a3"] + grand["stripped"]
    lines.append(
        f"| **GRAND TOTAL** | | | **{grand['a1']}** | "
        f"**{grand['a3']}** | **{grand['stripped']}** | |"
    )

    # Summary stats below the table.
    pct = (100 * grand["stripped"] / grand["attempted"]) if grand["attempted"] else 0
    lines.append("")
    lines.append("## Summary\n")
    lines.append(f"- Arm 1 (vanilla) total criterion codes cited: "
                 f"**{grand['a1']}** across all {len(results)} FAQs.")
    lines.append(f"- Arm 3 (full system) grounded criterion codes: **{grand['a3']}**.")
    lines.append(f"- Arm 3 attempted **{grand['attempted']}** criterion citations; "
                 f"the grounding filter stripped **{grand['stripped']}** as ungrounded "
                 f"(**{pct:.0f}%** hallucination rate).")

    out = "\n".join(lines)
    out_path = RESULTS_DIR / "results_table.md"
    out_path.write_text(out)
    print(out)
    print(f"\n\nSaved to: {out_path}")


if __name__ == "__main__":
    main()