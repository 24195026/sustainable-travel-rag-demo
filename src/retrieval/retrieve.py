"""Retrieval module for the Sustainable Travel RAG chatbot.

Public API:
    retrieve(query, k=5) -> list[RetrievedChunk]
    RetrievedChunk: typed dataclass with all chunk metadata

Connection to ChromaDB and the embedding model are cached at module level,
so the embedding weights load once per Python process.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

# Configuration constants
DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "sustainable_travel_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 5

# Phase 4.2.5.b: Query expansion via Llama-3.1-8B
EXPANDER_MODEL = "llama-3.1-8b-instant"

EXPANDER_PROMPT = """You are a search-query expander for a sustainable travel knowledge base containing GSTC criteria documents, Green Key criteria, EarthCheck standards, GDS-Index rankings, UNWTO statistics, UNWTO responsible-traveller guidance, the Climate Friendly Travel Registry, and the Green Destinations Collection (a list of specific destinations that hold Green Destinations Awards — Bronze/Silver/Gold/Platinum — and Top 100 Good Practice recognition).

Given a user query, output exactly 2 keyword-rich variants designed to surface criterion-style and standard-document text that consumer phrasings might miss.

Output rules:
- Return ONLY a JSON array of 2 strings. No preamble. No explanation.
- Each variant should be 6-12 words, dense with relevant keywords.
- Include specific scheme/criterion names where relevant (Green Key, EarthCheck, GSTC, hostels, audit, certification).
- One variant should lean toward accommodation/certification vocabulary.
- One variant should lean toward operator/activity vocabulary.

CRITICAL CONSTRAINT — countries and destinations:
You may use ONLY country, city, or region names that appear literally in the user\'s query. If the user query says \"Asia\", you may use \"Asia\". If the user query says \"Bali\", you may use \"Bali\". You may NOT add Laos, Cambodia, Vietnam, Thailand, Indonesia, or any other country or city not in the user\'s query. Variants that invent destination names break the retrieval system. Use generic terms like \"destinations\", \"regions\", \"low-carbon destinations\" instead of inventing specific places.

Example:
User query: \"Are there real sustainability certifications for budget accommodation?\"
Output: [\"Green Key EarthCheck hostel certification audit criteria budget accommodation\", \"GSTC accredited tour operator sustainability standard scheme\"]

Example with region in query:
User query: \"What is a sustainable destination in Asia with lower carbon footprint?\"
Output: [\"EarthCheck certified destinations Asia low-carbon sustainability standard\", \"GSTC accredited Asian destinations carbon emissions criteria audit\"]

DESTINATION-DISCOVERY queries:
When the user is asking WHERE to go (finding, choosing, or comparing destinations — e.g. dupes, alternatives, cool/uncrowded places), ONE variant should be dense in Green Destinations Collection vocabulary so it can match awarded-destination entries. Use phrases like "Green Destinations Collection awarded certified destination" plus any region word ALREADY in the user's query. Still obey the country/destination constraint above — do NOT invent specific place names; rely on the scheme/award vocabulary to surface them.

Example (destination discovery):
User query: "Somewhere cool and uncrowded for summer that won't be ruined by tourists"
Output: ["Green Destinations Collection awarded certified cool-climate destination low visitor numbers", "GSTC destination criteria visitor management overtourism climate adaptation"]"""

# Module-level cache for query expansion: query -> list of variant strings
_expand_cache: dict[str, list[str]] = {}


@dataclass
class RetrievedChunk:
    """One chunk returned from ChromaDB retrieval, with all metadata.

    Fields map directly to the chunk schema produced by chunk_pdfs.py and
    build_index.py. The `distance` field is the cosine distance from the
    query embedding; smaller = more similar.
    """
    chunk_id: str
    text: str
    source_id: int
    source_name: str
    tier: int
    tier_weight: float
    page_number: int
    chunk_index: int
    token_count: int
    faq_relevance: str
    distance: float


# Module-level cache for the collection. Loaded on first retrieve() call.
_collection: Optional[Collection] = None


def _get_collection() -> Collection:
    """Lazy-load the ChromaDB collection on first use, then cache it.

    The embedding model is ~91 MB and takes a few seconds to load. Caching
    avoids reloading on every retrieve() call, which matters for the
    Streamlit app and end-to-end evaluation runs.
    """
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedder,
        )
    return _collection


def retrieve(query: str, k: int = DEFAULT_K, max_per_source: int = 2) -> list[RetrievedChunk]:
    """Retrieve the top-k chunks with per-source diversity caps.

    Over-retrieves from ChromaDB then applies a per-source cap to prevent
    any single source from dominating the result list. For k=5 and
    max_per_source=2, no single source contributes more than 40% of the
    returned chunks.

    Args:
        query: Natural-language query string.
        k: Number of chunks to return (default 5).
        max_per_source: Maximum chunks from any single source (default 2).
            Set to None to disable the cap (pre-Phase-4.2.5 behaviour).

    Returns:
        List of RetrievedChunk in similarity order, deduplicated and
        capped at max_per_source per source. Length may be less than k
        if the collection has too few diverse sources.

    Example:
        >>> from retrieval import retrieve
        >>> results = retrieve("budget hostel certification", k=5)
        >>> sources = [c.source_id for c in results]
        >>> len(set(sources)) >= 3
        True
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    collection = _get_collection()

    # Over-retrieve to give the source-cap algorithm room to find diversity
    over_k = k * 3
    results = collection.query(query_texts=[query], n_results=over_k)

    candidates: list[RetrievedChunk] = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source_id=int(meta["source_id"]),
                source_name=str(meta["source_name"]),
                tier=int(meta["tier"]),
                tier_weight=float(meta["tier_weight"]),
                page_number=int(meta["page_number"]),
                chunk_index=int(meta["chunk_index"]),
                token_count=int(meta["token_count"]),
                faq_relevance=str(meta.get("faq_relevance", "")),
                distance=float(dist),
            )
        )

    # Apply per-source cap walking candidates in similarity order
    cap = max_per_source if max_per_source is not None else len(candidates)
    chunks: list[RetrievedChunk] = []
    seen_per_source: dict[int, int] = {}
    for chunk in candidates:
        count = seen_per_source.get(chunk.source_id, 0)
        if count < cap:
            chunks.append(chunk)
            seen_per_source[chunk.source_id] = count + 1
            if len(chunks) >= k:
                break

    return chunks


def expand_query(query: str) -> list[str]:
    """Use Groq Llama-3.1-8B to generate 2 keyword-rich query variants.

    Returns a list of 0 or 2 variants. On any failure (parse error,
    malformed output, API error), returns an empty list - callers should
    fall back to using just the original query.

    Result is cached in memory by lowercased query string. Identical
    queries in the same Python process do not re-hit the Groq API.

    Args:
        query: Natural-language query string.

    Returns:
        list[str]: 2 variant query strings, or [] on any failure.
    """
    if not query or not query.strip():
        return []

    cache_key = query.strip().lower()
    if cache_key in _expand_cache:
        return _expand_cache[cache_key]

    variants: list[str] = []
    try:
        import os
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        response = client.chat.completions.create(
            model=EXPANDER_MODEL,
            messages=[
                {"role": "system", "content": EXPANDER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        if (isinstance(parsed, list)
                and len(parsed) == 2
                and all(isinstance(v, str) and v.strip() for v in parsed)):
            variants = [v.strip() for v in parsed]
    except Exception:
        # Any failure (network, JSON parse, malformed response, key missing)
        # falls back to the empty list. Callers will use original query only.
        variants = []

    _expand_cache[cache_key] = variants
    return variants


def expand_and_retrieve(
    query: str,
    k: int = DEFAULT_K,
    max_per_source: int = 2,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks using LLM query expansion + per-source caps.

    Workflow:
        1. Call expand_query() to get 0-2 keyword-rich variants of the query.
        2. Retrieve for the original query AND each variant separately, each
           using the per-source cap.
        3. Merge all results, deduplicating by chunk_id, keeping the smallest
           distance seen for each chunk across all variants.
        4. Apply the per-source cap again on the merged result.
        5. Return top-k by distance.

    If expansion fails (returns 0 variants), this gracefully falls back to
    a normal retrieve() call — chatbot continues to work, just without the
    expansion benefit.

    Args:
        query: Natural-language query string.
        k: Number of chunks to return (default 5).
        max_per_source: Per-source cap, applied both in each retrieve() call
            and in the final merge step (default 2).

    Returns:
        List of RetrievedChunk in best-distance order, deduplicated and
        capped at max_per_source per source.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Try to expand. If expansion fails, fall back to normal retrieval.
    variants = expand_query(query)
    if not variants:
        return retrieve(query, k=k, max_per_source=max_per_source)

    # Retrieve for original + each variant
    all_queries = [query] + variants

    # Deduplicate by chunk_id, keeping the smallest distance seen
    best_by_chunk: dict[str, RetrievedChunk] = {}
    for q in all_queries:
        chunks = retrieve(q, k=k, max_per_source=max_per_source)
        for c in chunks:
            existing = best_by_chunk.get(c.chunk_id)
            if existing is None or c.distance < existing.distance:
                best_by_chunk[c.chunk_id] = c

    # Sort merged candidates by distance (ascending = best first)
    merged = sorted(best_by_chunk.values(), key=lambda c: c.distance)

    # Apply per-source cap one more time on the merged list
    cap = max_per_source if max_per_source is not None else len(merged)
    final: list[RetrievedChunk] = []
    seen_per_source: dict[int, int] = {}
    for c in merged:
        count = seen_per_source.get(c.source_id, 0)
        if count < cap:
            final.append(c)
            seen_per_source[c.source_id] = count + 1
            if len(final) >= k:
                break

    return final


if __name__ == "__main__":
    # Quick self-test when run as a script
    print("Self-test: retrieving for 'annual energy reduction targets for hotels'")
    results = retrieve("annual energy reduction targets for hotels", k=3)
    print(f"Got {len(results)} chunks")
    for i, c in enumerate(results, start=1):
        print(f"  {i}. {c.chunk_id} (Source {c.source_id}, page {c.page_number}, "
              f"distance {c.distance:.3f})")
        print(f"     {c.source_name} [Tier {c.tier}, weight {c.tier_weight}]")
