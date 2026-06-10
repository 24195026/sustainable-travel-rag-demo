"""Generation module for the Sustainable Travel RAG chatbot.

Public API:
    generate_response(messages) -> str
        Thin wrapper around the Groq Chat Completions API (llama-3.3-70b).
    answer(query, k=5) -> str
        Full pipeline: expand_and_retrieve -> build_messages -> generate.
        Query in, grounded answer out. Used by the Streamlit UI.
    answer_with_metadata(query, k=5) -> dict
        Same pipeline but returns response plus retrieved chunks,
        classification, and token usage for evaluation logging.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from groq import Groq

# Make sibling packages importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import RetrievedChunk
from retrieval.retrieve import expand_and_retrieve
from generation.prompts import build_messages, classify_query_llm
from generation.grounding import filter_ungrounded_codes
from reranking.reranker import rerank
from reranking.score import Weights

# Configuration
GENERATION_MODEL = "llama-3.3-70b-versatile"
GENERATION_TEMPERATURE = 0.3
GENERATION_SEED = 42
GENERATION_MAX_TOKENS = 1024

# Module-level cached Groq client
_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    """Lazy-init the Groq client on first use."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. "
                "Add it to your Codespace secrets and restart the Codespace."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate_response(messages: list[dict]) -> str:
    """Call the Groq generation model with a pre-built messages list.

    Args:
        messages: [{"role": "system", ...}, {"role": "user", ...}] as
            produced by prompts.build_messages().

    Returns:
        The model's response text (UK English, short inline citations).
    """
    if not messages:
        raise ValueError("messages list must not be empty")

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
        max_tokens=GENERATION_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def answer(
    query: str,
    k: int = 5,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    weights: Optional[Weights] = None,
) -> str:
    """End-to-end pipeline: query in, grounded answer out.

    Steps: expand_and_retrieve (k*2) -> rerank to top-k -> build_messages
    (classifies + picks variant) -> generate -> grounding filter.

    Over-retrieves by 2x then reranks to k, so the reranker can promote
    Tier-1 chunks over higher-similarity Tier-3 chunks. Origin and
    destination, when supplied, activate the carbon penalty.

    Args:
        query: Natural-language user query.
        k: Number of chunks ultimately passed to the LLM (default 5).
        origin: User's departure location (optional; for carbon penalty).
        destination: Trip destination (optional; for carbon penalty).
        weights: Reranker weights (default: Weights()).

    Returns:
        The chatbot's grounded response text.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    w = weights if weights is not None else Weights()
    candidates = expand_and_retrieve(query, k=k * 2)
    chunks = rerank(candidates, origin=origin, destination=destination, weights=w)[:k]
    if not chunks:
        return (
            "I could not find relevant sustainability sources for that query. "
            "Could you rephrase, or ask about a specific destination, "
            "accommodation type, or activity?"
        )
    messages = build_messages(query, chunks)
    raw = generate_response(messages)
    filtered, _ = filter_ungrounded_codes(raw, chunks)
    return filtered


def answer_with_metadata(
    query: str,
    k: int = 5,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    weights: Optional[Weights] = None,
) -> dict:
    """Same pipeline as answer() but returns rich metadata for evaluation.

    Returns a dict with:
        response: the chatbot answer text (post-grounding-filter)
        raw_response: the LLM output before the grounding filter
        stripped_codes: ungrounded criterion codes the filter removed
        category: "recommendation" or "verification"
        chunks: reranked chunks (top-k) used as context
        chunk_summary: list of (source_name, page, tier) tuples
        token_usage: dict with prompt/completion/total token counts
        weights_used: the Weights instance used (for Phase 6 grid logging)
        origin, destination: echoed back for evaluation provenance
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    w = weights if weights is not None else Weights()
    candidates = expand_and_retrieve(query, k=k * 2)
    chunks = rerank(candidates, origin=origin, destination=destination, weights=w)[:k]
    category = classify_query_llm(query)

    if not chunks:
        return {
            "response": (
                "I could not find relevant sustainability sources for that "
                "query. Could you rephrase, or ask about a specific "
                "destination, accommodation type, or activity?"
            ),
            "category": category,
            "chunks": [],
            "chunk_summary": [],
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        }

    messages = build_messages(query, chunks)

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
        max_tokens=GENERATION_MAX_TOKENS,
    )

    raw_response = response.choices[0].message.content or ""
    filtered_response, strip_log = filter_ungrounded_codes(raw_response, chunks)

    return {
        "response": filtered_response,
        "raw_response": raw_response,
        "stripped_codes": strip_log,
        "weights_used": w,
        "origin": origin,
        "destination": destination,
        "category": category,
        "chunks": chunks,
        "chunk_summary": [
            (c.source_name, c.page_number, c.tier) for c in chunks
        ],
        "token_usage": {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        },
    }


if __name__ == "__main__":
    test_query = (
        "I am a solo backpacker travelling Southeast Asia on a tight budget. "
        "I keep seeing hostels and guesthouses calling themselves eco-friendly "
        "but I cannot tell what is genuine. Are there real sustainability "
        "certifications for budget accommodation, and what should I look for?"
    )

    print("=" * 70)
    print("Phase 4.3 self-test: full pipeline on FAQ_011 query")
    print("=" * 70)
    print()
    print(f"QUERY: {test_query}")
    print()

    result = answer_with_metadata(test_query, k=5)

    print(f"CLASSIFICATION: {result['category']}")
    print()
    print("CHUNKS USED:")
    for name, page, tier in result["chunk_summary"]:
        print(f"  - {name} (page {page}, Tier {tier})")
    print()
    print("TOKEN USAGE:")
    tu = result["token_usage"]
    print(f"  prompt: {tu['prompt']}, completion: {tu['completion']}, total: {tu['total']}")
    print()
    print("=" * 70)
    print("CHATBOT RESPONSE")
    print("=" * 70)
    print(result["response"])
    print()
    print("=" * 70)
    print("Self-test complete.")
    print("=" * 70)
