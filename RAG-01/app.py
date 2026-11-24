import tempfile
from pathlib import Path

import streamlit as st
from openai import AuthenticationError, RateLimitError, APIConnectionError, NotFoundError

from src.config import OPENAI_API_KEY
from src.vectorstore import ingest_file, count_chunks, clear_collection
from src.rag import answer_question
from src.logger import logger

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")


def explain_error(error):
    """Turn technical errors into messages a user can act on."""
    if isinstance(error, AuthenticationError):
        return "API key rejected. Check the key in your .env file."
    if isinstance(error, RateLimitError):
        return "Rate limit reached or no API credit. Check your OpenAI billing."
    if isinstance(error, APIConnectionError):
        return "Could not reach OpenAI. Check your internet connection."
    if isinstance(error, NotFoundError):
        return "Model not found. Check CHAT_MODEL in config.py."
    logger.error("Unexpected error: (%s)", type(error).__name__)
    return f"Something went wrong: Check the local application log for details."


def show_sources(sources):
    """Show citations in a collapsible box under the answer."""
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['source']}, page {s['page']}** · similarity {s['score']}")
            st.caption(s["text"])


# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("📄 Documents")
    st.caption("Use only public, sample or synthetic documents.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading, chunking and embedding..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                for file in uploaded_files:
                    path = Path(tmp_dir) / file.name
                    path.write_bytes(file.getbuffer())
                    try:
                        n = ingest_file(path)
                        st.success(f"{file.name}: {n} chunk(s) added")
                    except ValueError as e:  # e.g. scanned PDF, empty file
                        st.error(f"{file.name}: {e}")
                    except Exception as e:
                        st.error(f"{file.name}: {explain_error(e)}")

    st.divider()
    st.metric("Chunks in knowledge base", count_chunks())

    if st.button("Clear knowledge base"):
        clear_collection()
        st.session_state.messages = []
        st.rerun()


# ---------- Main area: chat ----------
st.title("📚 Knowledge Assistant - [POC]")
st.caption("Ask questions about your documents. Every answer cites its sources, "
           "and the assistant says so when the answer isn't in the documents.")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far (needed because Streamlit reruns the script)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []))

# Handle a new question
if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if count_chunks() == 0:
            answer, sources = "Please upload and process a document first (see the sidebar).", []
        else:
            with st.spinner("Searching your documents..."):
                try:
                    result = answer_question(question)
                    answer, sources = result["answer"], result["sources"]
                except Exception as e:
                    answer, sources = f"⚠️ {explain_error(e)}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
