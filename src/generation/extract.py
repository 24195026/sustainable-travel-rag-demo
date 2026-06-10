"""LLM-based extraction of structured parameters from free-text queries.

Phase 5.3 of the Sustainable Travel RAG chatbot. The Streamlit chat UI
(Phase 5.2) accepts natural language; this module parses each user
message into the structured fields the rest of the engine needs:

    {"intent": "recommendation" | "verification",
     "origin": str | None,
     "destination": str | None}

Two LLM calls are made per query, by deliberate choice:
  1. classify_intent  -> recommendation / verification (binary)
  2. extract_entities -> {origin, destination} (JSON, optional fields)

Separating the calls makes each prompt small and focused, makes each
function independently testable and cacheable, and matches the Phase 4.2
classifier pattern that has been proven to work on FAQ_011.

Design rules:
- The extractor is FAITHFUL, not validating. It reports what the user
  said. Validation of place names is the geocoder's job (Phase 4.4.2),
  and the reranker's carbon term gracefully handles None (Phase 4.4.5).
- Every public function has a hardened fallback so the chatbot never
  crashes on a malformed LLM response: failure returns sensible defaults.
- Results are cached in memory by the (lowercased, stripped) query
  string so repeated identical queries do not re-hit Groq.

Public API:
    classify_intent(query) -> "recommendation" | "verification"
    extract_entities(query) -> {"origin": str | None,
                                "destination": str | None}
    extract_params(query)   -> {"intent": ..., "origin": ...,
                                "destination": ...}

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from groq import Groq

# Configuration
EXTRACT_MODEL = "llama-3.1-8b-instant"
EXTRACT_TEMPERATURE = 0.0  # deterministic - classification/extraction is
                           # a structured task, not a creative one.
EXTRACT_MAX_TOKENS = 200

# Module-level cached Groq client
_groq_client: Optional[Groq] = None

# In-memory caches keyed by the normalised query
_intent_cache: dict[str, str] = {}
_entity_cache: dict[str, dict] = {}


def _get_groq_client() -> Groq:
    """Lazy-init the Groq client on first use."""
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _normalise(query: str) -> str:
    return query.strip().lower()


# ---------------------------------------------------------------------------
# Function 1: classify_intent
# ---------------------------------------------------------------------------

_INTENT_SYSTEM = """You are a query classifier for a sustainable travel chatbot.

Classify each user query into EXACTLY ONE of two categories:

- "recommendation": the user wants suggestions about WHERE to go, what
  destinations to consider, or how to plan a trip. Examples:
    "Where should I go in Asia with a low carbon footprint?"
    "Suggest a sustainable beach destination from London."
    "I want a backpacker-friendly eco trip."

- "verification": the user wants to UNDERSTAND, CHECK, or LEARN about
  sustainability schemes, certifications, criteria, claims, or how to
  evaluate something. Examples:
    "What does GSTC Hotel A1 require?"
    "How do I know if a hostel is genuinely eco-friendly?"
    "What is Green Key certification?"

Output ONLY the single word "recommendation" or "verification".
No quotes, no punctuation, no explanation."""


def classify_intent(query: str) -> str:
    """Classify the query as 'recommendation' or 'verification'.

    Cached in memory by the lowercased query. Falls back to
    'verification' on any failure or on empty input (the safer default;
    the verification prompt template is more conservative about what it
    claims).
    """
    key = _normalise(query)
    # Early return: empty input gets the safe default without an API call.
    if not key:
        return "verification"
    if key in _intent_cache:
        return _intent_cache[key]

    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=EXTRACT_TEMPERATURE,
            max_tokens=20,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        # Strip trailing punctuation / quotes the model might add
        raw = raw.strip(' "\'.,!?')
        if raw not in ("recommendation", "verification"):
            raw = "verification"
    except Exception:
        raw = "verification"

    _intent_cache[key] = raw
    return raw


# ---------------------------------------------------------------------------
# Function 2: extract_entities
# ---------------------------------------------------------------------------

_ENTITY_SYSTEM = """You are an entity extractor for a sustainable travel chatbot.

From the user's message, extract the TRAVEL ORIGIN and TRAVEL DESTINATION
mentioned in the query, if any.

Rules:
1. Origin = where the user is travelling FROM (their home / departure point).
2. Destination = where the user is travelling TO, or the place they are
   asking about visiting.
3. If a field is NOT explicitly mentioned in the user's query, return null
   for that field. Do NOT invent or guess.
4. Use the place name AS WRITTEN by the user (do not normalise: keep
   "Bali" as "Bali", do not change to "Denpasar"; keep "Saigon" as
   "Saigon"; keep "NYC" as "NYC").
5. Country names are valid (e.g. "Germany" can be a destination).
6. Regions like "Southeast Asia" are valid as destination.
7. If MULTIPLE place names appear that could be destinations, return
   the one that appears FIRST in reading order (leftmost in the
   sentence). This applies regardless of whether the user is comparing
   them, listing them with "or"/"and", or referencing one as a benchmark
   ("better than X", "lower than X"). The reranker uses this single
   field; the full text of the user's message is still available to the
   chatbot downstream for prose comparisons.

   Worked examples (apply this rule strictly):

   User: "From London to somewhere in Asia, what destinations have lower carbon than Bali?"
   -> destination is "Asia" (first place after "to"); NOT "Bali" (mentioned later as a benchmark).

   User: "Should I go to Bali or Lombok?"
   -> destination is "Bali" (first mentioned).

   User: "Anywhere better than Bangkok?"
   -> destination is "Bangkok" (the only place mentioned).

Output EXACTLY this JSON object, with no additional text, no markdown
code fences, no commentary:

{"origin": "<place or null>", "destination": "<place or null>"}

Examples:

User: "I want to travel from Frankfurt to Bali"
Output: {"origin": "Frankfurt", "destination": "Bali"}

User: "What is Green Key certification?"
Output: {"origin": null, "destination": null}

User: "How can I find a sustainable hostel in Vietnam?"
Output: {"origin": null, "destination": "Vietnam"}

User: "From London to somewhere in Asia, low carbon"
Output: {"origin": "London", "destination": "Asia"}

User: "Is travelling from the Philippines to Germany sustainable?"
Output: {"origin": "Philippines", "destination": "Germany"}"""


def _safe_parse_entities(raw: str) -> dict:
    """Parse a JSON string into the entity dict; default on any failure."""
    default = {"origin": None, "destination": None}
    if not raw:
        return default
    # Strip code fences the model sometimes adds despite instructions
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    origin = data.get("origin")
    destination = data.get("destination")
    # Coerce empty strings / "null" strings / non-strings to None
    def _coerce(v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in ("null", "none", "n/a"):
                return None
            return s
        return None  # anything else (number, list) is unexpected
    return {"origin": _coerce(origin), "destination": _coerce(destination)}


def extract_entities(query: str) -> dict:
    """Extract {origin, destination} from a free-text query.

    Cached in memory. Falls back to {"origin": None, "destination": None}
    on any failure or on empty input - which the geocoder and reranker
    handle gracefully (carbon term goes to 0, exactly as designed for
    unknown-trip cases).
    """
    key = _normalise(query)
    # Early return: empty input gets safe defaults without an API call.
    if not key:
        return {"origin": None, "destination": None}
    if key in _entity_cache:
        return _entity_cache[key].copy()

    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL,
            messages=[
                {"role": "system", "content": _ENTITY_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=EXTRACT_TEMPERATURE,
            max_tokens=EXTRACT_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content or ""
        result = _safe_parse_entities(raw)
    except Exception:
        result = {"origin": None, "destination": None}

    _entity_cache[key] = result
    return result.copy()


# ---------------------------------------------------------------------------
# Function 3: extract_params (orchestrator)
# ---------------------------------------------------------------------------

def extract_params(query: str) -> dict:
    """Run both extractions and return a single combined dict.

    Returns:
        {"intent": "recommendation" | "verification",
         "origin": str | None,
         "destination": str | None}
    """
    intent = classify_intent(query)
    entities = extract_entities(query)
    return {
        "intent": intent,
        "origin": entities["origin"],
        "destination": entities["destination"],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (query, expected_intent, expected_origin_truthy, expected_dest_truthy)
        (
            "I want to travel from Frankfurt to Bali on a tight budget. "
            "What are low-carbon options?",
            "recommendation", True, True,
        ),
        (
            "From London to somewhere in Asia, what destinations have "
            "lower carbon than Bali?",
            "recommendation", True, True,
        ),
        (
            "What does Green Key certification actually verify?",
            "verification", False, False,
        ),
        (
            "Are hostels in Vietnam audited for sustainability?",
            "verification", False, True,
        ),
        (
            "I'm a backpacker, suggest a sustainable destination.",
            "recommendation", False, False,
        ),
        (
            "Is travelling from the Philippines to Germany sustainable?",
            "verification", True, True,
        ),
        (
            "What does GSTC Hotel A1 require?",
            "verification", False, False,
        ),
        (
            "What's the weather like today?",  # deliberately out-of-scope
            "verification", False, False,
        ),
        # --- Phase 5.3.2 edge cases ---
        # A1: empty string - must not crash
        (
            "",
            "verification", False, False,
        ),
        # A2: misspelled places - extractor must be FAITHFUL, not auto-correct
        (
            "From Frankfort to Filippines, what eco hotels are recommended?",
            "recommendation", True, True,
        ),
        # A3: ambiguous place (Cambridge UK vs US) - faithful pass-through
        (
            "I live in Cambridge and want a sustainable holiday.",
            "recommendation", True, False,
        ),
        # A4: non-ASCII diacritics
        (
            "From Köln to São Paulo, what are the low-carbon options?",
            "recommendation", True, True,
        ),
        # A5: multiple destinations - extractor has one field, will pick one
        (
            "Should I go to Bali or Lombok? Which is more sustainable?",
            "recommendation", False, True,
        ),
        # A6: multi-turn-style reference - no memory yet, treats as fresh
        (
            "And what about Tokyo?",
            "recommendation", False, True,
        ),
        # A7: long rambling query - extraction must stay focused
        (
            "I am a freelance designer based in Lisbon. I work remotely most "
            "of the time. I would love to take three weeks off in October and "
            "visit somewhere in Southeast Asia that takes sustainability "
            "seriously - what would you suggest?",
            "recommendation", True, True,
        ),
        # B3: off-topic but harmless - intent_required=False, just must not
        # invent travel parameters
        (
            "What's a good Italian recipe?",
            "verification", False, False, False,
        ),
        # C1: SunX / hotel-plugin scenario - verification, no locations
        (
            "Is this hotel really eco-friendly or just greenwashing?",
            "verification", False, False,
        ),
    ]

    print("Extractor self-test")
    print("=" * 70)
    passed = 0
    failed = 0
    for i, case in enumerate(test_cases, 1):
        # Tuple may be 4 or 5 elements (5th = intent_required, default True)
        q, exp_intent, exp_o, exp_d = case[:4]
        intent_required = case[4] if len(case) >= 5 else True

        result = extract_params(q)
        intent_match = (result["intent"] == exp_intent)
        intent_ok = intent_match or (not intent_required)
        origin_ok = (bool(result["origin"]) == exp_o)
        dest_ok = (bool(result["destination"]) == exp_d)
        all_ok = intent_ok and origin_ok and dest_ok
        mark = "PASS" if all_ok else "FAIL"
        if all_ok:
            passed += 1
        else:
            failed += 1
        print(f"[{mark}] case {i}: {q[:70]}")
        print(f"        intent={result['intent']:14s} "
              f"(expected {exp_intent})")
        print(f"        origin={str(result['origin']):20s} "
              f"(expected {'set' if exp_o else 'None'})")
        print(f"        destination={str(result['destination']):20s} "
              f"(expected {'set' if exp_d else 'None'})")
        print()
    print("=" * 70)
    print(f"OVERALL: {passed}/{passed+failed} passed")
