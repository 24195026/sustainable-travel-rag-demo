"""
Page-aware PDF chunker for the Sustainable Travel RAG corpus.

Reads each PDF in data/raw/, splits text into 512-token chunks (15% overlap)
respecting page boundaries, and attaches structured metadata for downstream
ChromaDB ingestion and reranker tier-weighting.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
Output:  data/processed/chunks.jsonl  (one JSON object per line, one per chunk)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import pypdf
import tiktoken

# ============================================================================
# CONFIGURATION
# ============================================================================
RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")
CHUNK_TOKENS = 512
OVERLAP_PCT = 0.15  # 15% overlap per project proposal
OVERLAP_TOKENS = int(CHUNK_TOKENS * OVERLAP_PCT)  # 76 tokens
MIN_CHUNK_TOKENS = 50  # discard chunks smaller than this (page artefacts)

# Encoder used purely for token counting and slicing — NOT for embeddings.
# cl100k_base is OpenAI's tokenizer; standard for length measurement.
ENCODER = tiktoken.get_encoding("cl100k_base")

# ============================================================================
# SOURCE REGISTRY (mirrors docs/sources_registry.md exactly)
# ============================================================================
# Schema: filename -> (source_id, source_name, tier, tier_weight, faqs_covered)
# Filenames must match data/raw/ exactly. Tier weights mirror sources_registry.md.

SOURCES = {
    "GSTC_Destination_Criteria_v2.0.pdf": (
        1, "GSTC Destination Criteria v2.0", 1, 1.5,
        ["001", "002", "003", "004", "005", "006", "007", "008",
         "009", "010", "011", "012"],
    ),
    "GSTC-Hotel-Standard.pdf": (
        2, "GSTC Hotel Standard v.4.0", 1, 1.5,
        ["004", "008", "009", "011", "012"],
    ),
    "GSTC-Tour-Operator-Standard.pdf": (
        3, "GSTC Tour Operator Standard v.4.0", 1, 1.5,
        ["005", "006", "008", "010", "011", "012"],
    ),
    "GSTC-Attraction-Criteria.pdf": (
        4, "GSTC Attraction Criteria v.1.0", 1, 1.5,
        ["003", "005", "008", "010", "012"],
    ),
    "Green_Key_Criteria.pdf": (
        5, "Green Key International Criteria 2023", 1, 1.5,
        ["004", "008", "009", "011"],
    ),
    "Earth Check.pdf": (
        6, "EarthCheck Benchmarking Standards 2022", 1, 1.5,
        ["005", "008", "010", "011"],
    ),
    "GDS-Index report.pdf": (
        7, "Global Destination Sustainability Index 2024", 2, 1.0,
        ["001", "004", "005", "011"],
    ),
    "UNWTO Tourism Highlights 2024.pdf": (
        8, "UNWTO Tourism Highlights 2024", 2, 1.0,
        ["002", "003", "005"],
    ),
    "Travalyst Carbon Emissions Framework Methodology.pdf": (
        9, "Travalyst Carbon Emissions Framework 2024", 2, 1.0,
        ["002", "012"],
    ),
    "OurWorldInData_Transport_CO2_2023.pdf": (
        10, "Our World in Data — CO2 Transport 2023", 3, 0.7,
        ["002", "006", "007"],
    ),
    "Booking_Sustainable_Travel_2024.pdf": (
        11, "Booking.com Sustainable Travel Report 2024", 3, 0.7,
        ["001", "004", "008"],
    ),
    "Responsible_Travel_Impact_2023.pdf": (
        12, "Responsible Travel Impact Report 2021-2023", 3, 0.7,
        ["003", "008", "010"],
    ),
    "CFT_Registry_About_2026.pdf": (
        13, "Climate Friendly Travel Registry (SUNx Malta)", 4, 0.4,
        ["001", "008", "011"],
    ),
    "GreenDestinations_Collection.pdf": (
        14, "Green Destinations Collection (GSTC-recognised)", 1, 1.5,
        ["001", "005", "006", "007"],
    ),
    "Tips-for-Responsible-Traveller-WCTE-EN.pdf": (
        15, "UNWTO Tips for a Responsible Traveler (WCTE)", 2, 1.0,
        ["012"],
    ),
    "CFT_Registry_Entries.pdf": (
        16, "Climate Friendly Travel Registry — Entries (SUNx Malta)", 4, 0.4,
        ["001", "004", "008", "011"],
    ),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class Chunk:
    """One retrievable unit. Maps directly to a ChromaDB record."""
    chunk_id: str            # e.g. "src01_p007_c00"
    source_id: int
    source_name: str
    tier: int
    tier_weight: float
    page_number: int         # 1-indexed page number
    chunk_index: int         # sequential index within this source
    token_count: int
    faq_relevance: list[str]
    text: str


# ============================================================================
# PDF -> PAGE TEXT
# ============================================================================
def extract_pages(pdf_path: Path) -> list[str]:
    """Return one string per page, in document order. Empty pages kept as ''."""
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"  WARN: extract failed on a page in {pdf_path.name}: {exc}")
            text = ""
        # Light cleanup: collapse runs of whitespace, strip page-edge noise
        text = re.sub(r"\s+", " ", text).strip()
        pages.append(text)
    return pages


# ============================================================================
# TOKEN-BASED CHUNKING (page-aware)
# ============================================================================
def chunk_page(
    text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> Iterator[tuple[str, int]]:
    """
    Split a single page's text into overlapping token windows.
    Yields (chunk_text, token_count) tuples.

    Page-aware: never crosses page boundaries. Pages with fewer than
    `min_tokens` tokens after extraction are skipped (covers blank pages,
    section dividers, image-heavy spreads).
    """
    if not text:
        return
    token_ids = ENCODER.encode(text)
    if len(token_ids) < min_tokens:
        return

    step = chunk_tokens - overlap_tokens  # 512 - 76 = 436
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_tokens, len(token_ids))
        window = token_ids[start:end]
        # Discard final tail if too short (avoids tiny dangling chunks)
        if len(window) < min_tokens and start > 0:
            break
        chunk_text = ENCODER.decode(window)
        yield chunk_text, len(window)
        if end == len(token_ids):
            break
        start += step


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def process_source(pdf_path: Path) -> list[Chunk]:
    meta = SOURCES.get(pdf_path.name)
    if meta is None:
        print(f"  SKIP: {pdf_path.name} not in SOURCES registry")
        return []

    source_id, source_name, tier, tier_weight, faqs = meta
    chunks: list[Chunk] = []
    chunk_index = 0

    pages = extract_pages(pdf_path)
    for page_num, page_text in enumerate(pages, start=1):
        for chunk_text, token_count in chunk_page(page_text):
            chunk_id = f"src{source_id:02d}_p{page_num:04d}_c{chunk_index:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    source_name=source_name,
                    tier=tier,
                    tier_weight=tier_weight,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    faq_relevance=faqs,
                    text=chunk_text,
                )
            )
            chunk_index += 1

    print(f"  {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks")
    return chunks


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(f"ERROR: {RAW_DIR} does not exist. Run from project root.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: list[Chunk] = []

    print(f"Chunking {len(SOURCES)} source PDFs from {RAW_DIR}/")
    print(f"Settings: {CHUNK_TOKENS} tokens, {OVERLAP_PCT:.0%} overlap, "
          f"page-aware, min {MIN_CHUNK_TOKENS} tokens/chunk")
    print()

    for filename in SOURCES:
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            print(f"  MISSING: {filename}")
            continue
        all_chunks.extend(process_source(pdf_path))

    # Write JSONL: one chunk per line. Easy to stream into ChromaDB later.
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print(f"Total chunks written: {len(all_chunks)}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    # Per-tier breakdown for sanity-checking against sources_registry.md
    by_tier: dict[int, int] = {}
    for c in all_chunks:
        by_tier[c.tier] = by_tier.get(c.tier, 0) + 1
    print("\nChunks by tier:")
    for tier in sorted(by_tier):
        print(f"  Tier {tier}: {by_tier[tier]:>5} chunks")


if __name__ == "__main__":
    main()

