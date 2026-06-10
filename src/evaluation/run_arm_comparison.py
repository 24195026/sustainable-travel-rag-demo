"""Phase 6 (slice): Arm 1 (vanilla) vs Arm 3 (full system) comparison.

Runs a small set of evaluation FAQs through both the vanilla-LLM baseline
and the full RAG system, then measures grounding-related signals that need
no external judge library:

  - criterion codes cited (A-D + 1-2 digits), using the SAME regex the
    grounding filter uses, so counts are consistent across both arms
  - certification-scheme names mentioned (to surface fabrications like the
    vanilla model's invented "STEF / ASEAN Tourism Standards")
  - ungrounded criterion codes (Arm 3 only): pulled directly from the
    grounding filter's stripped_codes - the proportion/count of codes the
    LLM cited that were NOT in the retrieved context

Full raw responses are saved to eval/results/ for manual quality review and
for quoting in the report. A summary table is printed to the terminal.

This is the 3-arm ablation's first two arms on a small slice. Scaling to all
12 FAQs and adding Arm 2 (basic RAG) is a matter of widening the loop; the
measurement code is unchanged. RAGAS faithfulness scoring is deferred to a
later session (dependency-version resolution required).

Usage (from project root):
    uv run python src/evaluation/run_arm_comparison.py

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sibling packages importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.generate import answer_with_metadata
from generation.grounding import _CODE_PATTERN
from evaluation.baseline import baseline_answer

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = PROJECT_ROOT / "eval" / "queries.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
N_FAQS = 12  # full evaluation set (all FAQs in queries.json)
DELAY_BETWEEN_FAQS = 4  # seconds; spaces out Groq calls to respect free-tier rate limits

# Certification schemes we expect (real ones) plus a note on fabrications.
# Lower-cased substring match. The point is descriptive, not exhaustive.
KNOWN_SCHEMES = [
    "gstc",
    "green key",
    "earthcheck",
    "travelife",
    "green globe",
    "rainforest alliance",
    "biosphere",
    "fair trade tourism",
]


def count_codes(text: str) -> list[str]:
    """Return all criterion codes (e.g. D1, A4, D12) cited in the text,
    using the grounding filter's own pattern for consistency."""
    return _CODE_PATTERN.findall(text or "")


def count_schemes(text: str) -> list[str]:
    """Return which known certification schemes are mentioned (lower-cased
    substring match)."""
    low = (text or "").lower()
    return [s for s in KNOWN_SCHEMES if s in low]


def evaluate_faq(faq: dict) -> dict:
    """Run one FAQ through both arms and collect grounding signals."""
    query = faq["query"]

    # --- Arm 1: vanilla LLM, no retrieval, no grounding ---
    arm1 = baseline_answer(query)
    arm1_codes = count_codes(arm1["response"])
    arm1_schemes = count_schemes(arm1["response"])

    # --- Arm 3: full system (retrieve + rerank + generate + grounding) ---
    arm3 = answer_with_metadata(query)
    arm3_codes = count_codes(arm3["response"])
    arm3_schemes = count_schemes(arm3["response"])
    # Ungrounded codes the filter stripped before the user saw them.
    arm3_stripped = arm3.get("stripped_codes", [])

    return {
        "id": faq["id"],
        "region": faq.get("region"),
        "type": faq.get("type"),
        "query": query,
        "expected_bias_signal": faq.get("expected_bias_signal"),
        "source_grounding_required": faq.get("source_grounding_required", []),
        "arm1_baseline": {
            "response": arm1["response"],
            "codes_cited": arm1_codes,
            "n_codes": len(arm1_codes),
            "schemes_mentioned": arm1_schemes,
            "token_usage": arm1["token_usage"],
        },
        "arm3_full_system": {
            "response": arm3["response"],
            "codes_cited": arm3_codes,
            "n_codes": len(arm3_codes),
            "schemes_mentioned": arm3_schemes,
            "ungrounded_codes_stripped": arm3_stripped,
            "n_stripped": len(arm3_stripped),
            "chunks_used": arm3.get("chunk_summary", []),
            "token_usage": arm3["token_usage"],
        },
    }


def main() -> None:
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    faqs = data["queries"][:N_FAQS]

    print("=" * 72)
    print(f"Phase 6 slice: Arm 1 (vanilla) vs Arm 3 (full system)  -  {len(faqs)} FAQs")
    print("=" * 72)

    results = []
    failed = []
    for i, faq in enumerate(faqs, 1):
        print(f"\n[{i}/{len(faqs)}] {faq['id']} ({faq.get('region')}, {faq.get('type')})")
        print(f"  Q: {faq['query'][:80]}...")
        try:
            r = evaluate_faq(faq)
            results.append(r)

            a1 = r["arm1_baseline"]
            a3 = r["arm3_full_system"]
            print(f"  Arm 1 (vanilla): codes={a1['n_codes']}, schemes={a1['schemes_mentioned']}")
            print(f"  Arm 3 (full):    codes={a3['n_codes']}, schemes={a3['schemes_mentioned']}, "
                  f"ungrounded stripped={a3['n_stripped']} {a3['ungrounded_codes_stripped']}")
        except Exception as e:
            # One failed FAQ (e.g. a Groq rate-limit) must not lose the rest.
            print(f"  !! FAILED: {type(e).__name__}: {e}")
            failed.append({"id": faq["id"], "error": f"{type(e).__name__}: {e}"})

        # Space out calls to respect Groq free-tier rate limits.
        if i < len(faqs):
            time.sleep(DELAY_BETWEEN_FAQS)

    # --- Save full results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"arm_comparison_{stamp}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "generated": stamp,
                "n_faqs": len(faqs),
                "n_succeeded": len(results),
                "n_failed": len(failed),
                "failed": failed,
                "arms": ["arm1_baseline_vanilla", "arm3_full_system"],
                "note": "Grounding-signal comparison. RAGAS faithfulness pending.",
                "results": results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    total_a1_codes = sum(r["arm1_baseline"]["n_codes"] for r in results)
    total_a3_codes = sum(r["arm3_full_system"]["n_codes"] for r in results)
    total_stripped = sum(r["arm3_full_system"]["n_stripped"] for r in results)
    print(f"FAQs succeeded: {len(results)}/{len(faqs)}   failed: {len(failed)}")
    if failed:
        print(f"  Failed: {[x['id'] for x in failed]}")
    print(f"Total criterion codes cited - Arm 1: {total_a1_codes}, Arm 3: {total_a3_codes}")
    print(f"Total ungrounded codes stripped by filter (Arm 3): {total_stripped}")
    print(f"\nFull responses saved to: {out_path}")

if __name__ == "__main__":
    main()