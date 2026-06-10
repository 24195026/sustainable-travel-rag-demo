"""Grounding verification layer for the Sustainable Travel RAG chatbot.

Post-processes LLM responses to strip criterion codes that are not
grounded in the retrieved context. This is a deterministic guardrail
against few-shot example leakage, where the LLM cites plausible-but-
ungrounded criterion codes (e.g., GSTC Hotel D1) recalled from the
prompt examples rather than from the chunks retrieved for the query.

Public API:
    filter_ungrounded_codes(response, chunks) -> (filtered_text, strip_log)

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import RetrievedChunk

# Matches a GSTC criterion code: a single capital letter A-D followed by
# 1-2 digits, e.g. A1, A4, A13, D1, D2, D12, B3, C3.
# Used to find the bare code inside a citation like "GSTC Hotel D1".
_CODE_PATTERN = re.compile(r"\b([A-D]\d{1,2})\b")

# Matches a full GSTC citation inside parentheses, capturing the scheme
# label and the code, e.g. "GSTC Hotel D1", "GSTC Tour Op D12".
# Group 1 = scheme label ("GSTC Hotel"), Group 2 = code ("D1").
_GSTC_CITATION_PATTERN = re.compile(
    r"(GSTC(?:\s+(?:Hotel|Tour\s+Op|Tour\s+Operator|Attraction|Destination))?)\s+([A-D]\d{1,2})\b"
)


def _code_in_chunks(code: str, chunk_text: str) -> bool:
    """Return True if the bare code appears as a whole token in the text.

    Uses word boundaries so 'D1' does not match inside 'D12'.
    """
    pattern = re.compile(r"\b" + re.escape(code) + r"\b")
    return bool(pattern.search(chunk_text))


def filter_ungrounded_codes(
    response: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, list[dict]]:
    """Strip GSTC criterion codes not grounded in the retrieved chunks.

    For each "GSTC <Scheme> <Code>" citation in the response, checks
    whether <Code> appears as a whole token in any retrieved chunk's
    text. If not, the specific code is removed and the citation is
    downgraded to the general scheme name (e.g., "GSTC Hotel D1" ->
    "GSTC Hotel Standard").

    Args:
        response: The raw LLM response text.
        chunks: The RetrievedChunk objects used as context.

    Returns:
        (filtered_text, strip_log) where strip_log is a list of dicts
        recording each code that was stripped, for evaluation metrics.
    """
    all_chunk_text = "\n".join(c.text for c in chunks)
    strip_log: list[dict] = []

    def _replace(match: re.Match) -> str:
        scheme = match.group(1).strip()
        code = match.group(2)
        if _code_in_chunks(code, all_chunk_text):
            # Grounded - keep as-is
            return match.group(0)
        # Ungrounded - downgrade to scheme name, log the strip
        strip_log.append({"scheme": scheme, "code": code})
        # "GSTC Hotel" -> "GSTC Hotel Standard"; bare "GSTC" -> "GSTC Standard"
        if scheme == "GSTC":
            return "GSTC Standard"
        return f"{scheme} Standard"

    filtered = _GSTC_CITATION_PATTERN.sub(_replace, response)
    return filtered, strip_log


if __name__ == "__main__":
    # Self-test with a synthetic example
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        text: str

    # Chunks contain A1 and A2 but NOT A4, D1, or D2
    fake_chunks = [
        FakeChunk(text="A1 Sustainability Management System. The hotel operates..."),
        FakeChunk(text="A2 Legal Compliance. The hotel complies with..."),
    ]

    sample = (
        "Look for a sustainability management system (GSTC Hotel A1) and "
        "legal compliance (GSTC Hotel A2). Ask about energy targets "
        "(GSTC Hotel D1) and water conservation (GSTC Hotel D2). Reports "
        "should be published (GSTC Hotel A4)."
    )

    print("BEFORE:")
    print(sample)
    print()

    filtered, log = filter_ungrounded_codes(sample, fake_chunks)

    print("AFTER:")
    print(filtered)
    print()
    print(f"STRIPPED {len(log)} ungrounded code(s):")
    for entry in log:
        print(f"  - {entry['scheme']} {entry['code']}")
    print()
    print("Expected: A1 and A2 kept (grounded); D1, D2, A4 stripped (ungrounded)")
