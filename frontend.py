"""Local browser interface for testing the lunar RAG pipeline."""

import streamlit as st

from corpus import SOURCES
from generation import answer
from search import retrieve

st.set_page_config(page_title="Lunar research desk", page_icon="🌘")
st.title("Lunar research desk")
st.caption(
    "Ask a standalone question about the Moon and inspect the NASA passages behind the answer."
)

history = st.session_state.setdefault("history", [])
if st.sidebar.button("Clear conversation"):
    history.clear()


def show_turn(turn):
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
        if turn.get("answer"):
            st.markdown(turn["answer"])
        if turn["sources"]:
            with st.expander("Inspect all retrieved passages", expanded=False):
                for rank, row in enumerate(turn["sources"], 1):
                    title = {"moon": "Moon lithograph", "guide": "Exploring the Moon"}[
                        row["source"]
                    ]
                    url = SOURCES[row["source"]]
                    st.markdown(
                        f"**{rank}. [{title} · page {row['page']}]({url}#page={row['page']})**"
                    )
                    st.text(row["body"])


for turn in history:
    show_turn(turn)

question = st.chat_input("Ask about the Moon", max_chars=1500)
if question and question.strip():
    turn = {"question": question.strip(), "sources": []}
    with st.spinner("Searching NASA publications…"):
        try:
            rows = retrieve(turn["question"])
            # Keep only display data in browser-session history.
            turn["sources"] = [
                {key: row[key] for key in ("source", "page", "body")} for row in rows
            ]
        except Exception as error:
            turn["error"] = (
                f"Search failed ({type(error).__name__}). Check PostgreSQL, the passage index, and local model availability."
            )
    if turn["sources"]:
        with st.spinner("Checking the answer against the retrieved passages…"):
            try:
                turn["answer"] = answer(turn["question"], turn["sources"])
            except Exception as error:
                turn["error"] = (
                    f"Generation failed ({type(error).__name__}). Check the Groq model, credentials, and rate limits. You can still inspect the passages."
                )
    history.append(turn)
    show_turn(turn)
