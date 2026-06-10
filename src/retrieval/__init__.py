"""Retrieval module for the Sustainable Travel RAG chatbot.

Exports:
    retrieve(query, k=5) -> list[RetrievedChunk]
    RetrievedChunk dataclass with typed fields
"""
from .retrieve import retrieve, RetrievedChunk

__all__ = ["retrieve", "RetrievedChunk"]
