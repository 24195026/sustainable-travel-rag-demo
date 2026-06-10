"""Arm 1 (vanilla LLM) baseline for the Phase 6 ablation.

The honest "why not just use a plain LLM?" comparator. Sends the user's
question to the SAME Groq model as the full system (llama-3.3-70b-versatile)
with the SAME decoding settings (temperature, seed, max_tokens), but with:
    - NO retrieval (no context from the sustainability corpus)
    - NO grounding filter (no criterion-code hallucination guard)
    - NO reranker
    - a NEUTRAL, generic travel-assistant system prompt (NOT the project's
      sustainability-grounded prompt)

This isolation is deliberate and is the basis of a fair ablation: the only
variable that differs between Arm 1 and Arm 3 (the full system) is the
presence of Retrieval-Augmented Generation. Giving the baseline the project's
grounded system prompt minus the context would let it inherit the project's
sustainability instructions and would understate RAG's measured contribution.
The neutral prompt represents what an everyday user receives from an
ungrounded chatbot.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling packages importable (mirrors generate.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.generate import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    GENERATION_SEED,
    GENERATION_MAX_TOKENS,
    _get_groq_client,
)

# A deliberately generic, neutral system prompt. No sustainability coaching,
# no citation instructions, no framework awareness - this is the "plain
# chatbot" any user would get without the project's grounding.
_BASELINE_SYSTEM = (
    "You are a helpful travel assistant. Answer the user's travel question "
    "clearly and concisely."
)


def baseline_answer(query: str) -> dict:
    """Arm 1: vanilla LLM response, no retrieval, no grounding.

    Args:
        query: The natural-language user query (the same FAQ text passed
            to the full system).

    Returns:
        dict with:
            response: the model's answer text
            category: always "baseline" (no classification step in Arm 1)
            chunks: always [] (no retrieval in Arm 1)
            chunk_summary: always [] (no retrieval in Arm 1)
            token_usage: prompt/completion/total token counts
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _BASELINE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
        max_tokens=GENERATION_MAX_TOKENS,
    )

    return {
        "response": response.choices[0].message.content or "",
        "category": "baseline",
        "chunks": [],
        "chunk_summary": [],
        "token_usage": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }


if __name__ == "__main__":
    # Quick self-test: run one FAQ-style query through the baseline.
    test_query = (
        "I am a solo backpacker travelling Southeast Asia on a tight budget. "
        "Are there real sustainability certifications for budget "
        "accommodation, and what should I look for?"
    )
    print("=" * 70)
    print("Arm 1 (vanilla LLM) baseline self-test")
    print("=" * 70)
    result = baseline_answer(test_query)
    print(f"\nQUERY: {test_query}\n")
    print(f"CATEGORY: {result['category']}")
    print(f"TOKENS: {result['token_usage']}")
    print("\nRESPONSE:\n")
    print(result["response"])