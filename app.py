"""Sustainable Travel RAG Chatbot - Streamlit chat UI.

A free-text chat interface for the chatbot built in Phase 4. Users type
questions in natural language; the system retrieves grounded answers
from the curated sustainability corpus and renders them with inline
citations.

This file replaces the Phase 5.1 form skeleton. The form approach was
deliberately abandoned during the 29 May reflection: it served research
evaluation but worked against the chatbot premise and against
downstream deployment (SunX Malta and the paid hotel-plugin pathway).

Phase 5.2 scope: chat shell with visible history, single-turn responses
(no memory between turns yet - that is a Phase 5.5/5.6 concern), no
natural-language parameter extraction yet (Phase 5.3), no web fallback
yet (Phase 5.4).

Run from the project root:
    streamlit run app.py

Author: Bonna Bambilla, MSc Data Science, Arden University
Project: COM7014 Advanced Computing Project
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make project src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ---------- Backend (Phase 5.3.3 wiring) ----------
# Imported after src/ is on the path above, so these resolve.
from generation.extract import extract_params
from generation.generate import answer_with_metadata

from retrieval.retrieve import _get_collection


# ---------- Engine warm-up (Phase 5.3.3 performance fix) ----------
# Streamlit re-runs this whole script on every interaction. Without
# caching, the ChromaDB connection and the 91 MB sentence-transformers
# embedding model would reload each time (~9s measured cold-start). This
# caused long waits on every message. @st.cache_resource holds the warmed
# engine in memory across all reruns and sessions, so the load happens
# exactly once - on first page visit. _get_collection() is the single
# entry point that both connects ChromaDB and loads the embedder.
@st.cache_resource(show_spinner="Loading the sustainability engine (first load only)...")
def _warm_engine():
    _get_collection()
    return True


_warm_engine()

# ---------- Page config ----------
st.set_page_config(
    page_title="Sustainable Travel Assistant",
    page_icon=":earth_africa:",
    layout="centered",
)

# ---------- Arden University branding ----------
ARDEN_NAVY = "#002B4F"
ARDEN_TEAL = "#65C1BE"
ARDEN_YELLOW = "#FCBF00"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: #FFFFFF; }}
      h1 {{ color: {ARDEN_NAVY}; font-weight: 800; }}
      [data-testid="stSidebar"] {{ background-color: #F4F8F8; }}
      [data-testid="stSidebar"] h3 {{ color: {ARDEN_NAVY}; }}
      .stChatMessage {{ border-radius: 10px; }}
      div.stButton > button {{
        background: {ARDEN_TEAL}; color: {ARDEN_NAVY};
        border: none; font-weight: 600;
      }}
      div.stButton > button:hover {{ background: {ARDEN_NAVY}; color: #FFFFFF; }}
      a {{ color: {ARDEN_TEAL}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

import os as _os
_logo = "assets/arden_logo.png"
_c1, _c2 = st.columns([1, 6])
with _c1:
    if _os.path.exists(_logo):
        st.image(_logo, width=72)
with _c2:
    st.markdown(
        f"<div style='padding-top:6px'><span style='color:{ARDEN_NAVY};"
        f"font-weight:800;font-size:1.05rem'>Arden University</span><br>"
        f"<span style='color:#5a6b76;font-size:0.85rem'>MSc Data Science · "
        f"COM7014 Advanced Computing Project</span></div>",
        unsafe_allow_html=True,
    )

st.title("Sustainable Travel Assistant")
st.caption(
    "Ask anything about sustainable travel: certifications, low-carbon "
    "destinations, greenwashing red flags, eco-friendly accommodation. "
    "Answers are grounded in audited frameworks (GSTC, Green Key, "
    "EarthCheck) and authoritative sources (UNWTO, OurWorldInData, "
    "Travalyst)."
)

# ---------- Conversation state ----------
# Each message is a dict: {"role": "user" | "assistant", "content": str}
# History is shown in the UI but not yet passed to the LLM between turns
# (single-turn responses for Phase 5.2; multi-turn memory is Phase 5.5+).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: minimal controls
with st.sidebar:
    st.subheader("About")
    st.markdown(
        "A closed-corpus retrieval-augmented (RAG) assistant. It answers "
        "only from a curated knowledge base of 16 audited sustainability "
        "sources, citing each claim inline. Questions outside that scope "
        "receive an honest 'not in my sources' redirect rather than a guess."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Render conversation history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
# st.chat_input renders a sticky input box at the bottom of the page.
user_text = st.chat_input("Ask about sustainable travel...")

if user_text:
    # Record the user message and render it immediately.
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Phase 5.3.3: real backend. Extract origin/destination from the raw
    # message, then call the RAG engine. extract_params also classifies
    # intent internally, but answer_with_metadata runs its own (cached)
    # classification - we use extract_params here only for origin/destination,
    # which drive the reranker's carbon penalty.
    with st.chat_message("assistant"):
        with st.spinner("Searching the sustainability corpus..."):
            try:
                params = extract_params(user_text)
                result = answer_with_metadata(
                    user_text,
                    origin=params["origin"],
                    destination=params["destination"],
                )
                reply = result["response"]
            except Exception as e:
                reply = (
                    "Sorry - something went wrong while answering. "
                    "Please try again in a moment.\n\n"
                    f"_(Error: {type(e).__name__})_"
                )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------- Footer ----------
st.divider()
st.caption(
    "Research prototype. Recommendations are evidence-based but not a "
    "substitute for professional travel advice. Sources cited inline."
)

