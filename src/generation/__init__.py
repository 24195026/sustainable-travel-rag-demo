"""Generation module for the Sustainable Travel RAG chatbot.

Exports:
    SYSTEM_PROMPT_RECOMMENDATION, SYSTEM_PROMPT_VERIFICATION (str constants)
    classify_query_llm(query) -> "recommendation" | "verification"
    build_messages(query, chunks) -> list of {"role", "content"}
    format_chunks_for_prompt(chunks) -> str
    generate_response(messages) -> str
    answer(query, k=5) -> str
    answer_with_metadata(query, k=5) -> dict
    filter_ungrounded_codes(response, chunks) -> (str, list)
"""
from .prompts import (
    SYSTEM_PROMPT_RECOMMENDATION,
    SYSTEM_PROMPT_VERIFICATION,
    classify_query_llm,
    build_messages,
    format_chunks_for_prompt,
)
from .generate import (
    generate_response,
    answer,
    answer_with_metadata,
)
from .grounding import filter_ungrounded_codes

__all__ = [
    "SYSTEM_PROMPT_RECOMMENDATION",
    "SYSTEM_PROMPT_VERIFICATION",
    "classify_query_llm",
    "build_messages",
    "format_chunks_for_prompt",
    "generate_response",
    "answer",
    "answer_with_metadata",
    "filter_ungrounded_codes",
]
