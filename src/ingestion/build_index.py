"""
ChromaDB indexer for the Sustainable Travel RAG corpus.

Reads chunks from data/processed/chunks.jsonl, embeds each chunk using
sentence-transformers/all-MiniLM-L6-v2, and writes them to a persistent
ChromaDB collection with full metadata preserved.

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
Output:  data/chroma_db/  (persistent ChromaDB store)
"""
from __future__ import annotations
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "sustainable_travel_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(path):
    if not path.exists():
        raise SystemExit(f"ERROR: {path} does not exist. Run chunk_pdfs.py first.")
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        raise SystemExit(f"ERROR: {path} is empty.")
    return chunks


def flatten_metadata(chunk):
    return {
        "source_id": chunk["source_id"],
        "source_name": chunk["source_name"],
        "tier": chunk["tier"],
        "tier_weight": chunk["tier_weight"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "token_count": chunk["token_count"],
        "faq_relevance": ",".join(chunk["faq_relevance"]),
    }


def main():
    print(f"Loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")
    print()
    print(f"Initialising ChromaDB at {DB_PATH}")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(first run will download ~80 MB; subsequent runs are instant)")
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    print(f"Creating collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Embedding and adding {len(chunks)} chunks in batches of {BATCH_SIZE}")
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [flatten_metadata(c) for c in batch]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        progress = min(batch_start + BATCH_SIZE, len(chunks))
        print(f"  added {progress}/{len(chunks)}")
    final_count = collection.count()
    print()
    print("=" * 60)
    print(f"Indexing complete. Collection holds {final_count} chunks.")
    print(f"Persistent store: {DB_PATH}")
    print("=" * 60)
    print()
    print("Smoke test: querying for a known concept")
    results = collection.query(
        query_texts=["energy reduction targets for hotels"],
        n_results=3,
    )
    for i, (chunk_id, meta, doc) in enumerate(zip(
        results["ids"][0],
        results["metadatas"][0],
        results["documents"][0],
    ), start=1):
        print(f"\n  Hit {i}: {chunk_id}")
        sn = meta["source_name"]; tr = meta["tier"]; tw = meta["tier_weight"]
        pg = meta["page_number"]; fq = meta["faq_relevance"]
        print(f"    source: {sn} (tier {tr}, weight {tw})")
        print(f"    page {pg}, FAQs: {fq}")
        print(f"    preview: {doc[:140]}...")


if __name__ == "__main__":
    main()
