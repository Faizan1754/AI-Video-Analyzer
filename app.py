# app.py
"""AI Meeting Assistant — Streamlit front end.

Analyzes YouTube videos and local recordings: transcription, translation,
summarization, and RAG-based Q&A over the transcript.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

APP_TITLE = "AI Meeting Assistant"
APP_ICON = "🎙️"
MAX_UPLOAD_MB = 500
YOUTUBE_URL_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
)
SUPPORTED_FORMATS = ["mp4", "mov", "avi", "mkv", "m4a", "mp3", "wav"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Header */
        .app-header { display: flex; align-items: center; gap: .6rem; margin-bottom: 0; }
        .app-subtitle { color: var(--text-color, #8a8f98); font-size: 0.95rem; margin-top: -0.4rem; }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.06);
            border: 1px solid rgba(127, 127, 127, 0.15);
            border-radius: 10px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 8px;
            font-weight: 500;
        }

        /* Chat */
        .stChatMessage { border-radius: 10px; }

        /* Sidebar section labels */
        .sidebar-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(127,127,127,0.8);
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
        }

        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


@dataclass
class SessionState:
    result: dict | None = None
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    temp_path: str | None = None
    processing: bool = False


def get_state() -> SessionState:
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


state = get_state()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown(
    f'<div class="app-header"><h1>{APP_ICON} {APP_TITLE}</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Transcribe, translate, and query meeting recordings '
    "or YouTube videos with local speech-to-text and RAG-powered chat.</p>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------- #
# Sidebar — inputs
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.caption("Local-first meeting analysis")

    st.markdown('<p class="sidebar-label">Source</p>', unsafe_allow_html=True)
    input_type = st.radio(
        "Input source", ["YouTube URL", "Upload File"], label_visibility="collapsed"
    )

    source = None
    uploaded_file = None

    if input_type == "YouTube URL":
        url = st.text_input(
            "YouTube URL", placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        if url and not YOUTUBE_URL_PATTERN.match(url.strip()):
            st.caption("⚠️ That doesn't look like a valid YouTube URL.")
        source = url.strip() if url else None
    else:
        uploaded_file = st.file_uploader(
            "Upload a recording",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed",
            help=f"Max {MAX_UPLOAD_MB} MB. Video or audio files.",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                st.error(f"File is {size_mb:.0f} MB — exceeds {MAX_UPLOAD_MB} MB limit.")
                uploaded_file = None

    st.markdown('<p class="sidebar-label">Options</p>', unsafe_allow_html=True)
    language = st.selectbox("Output language", ["english", "hinglish"])

    st.write("")
    analyze = st.button(
        "🚀 Analyze", use_container_width=True, type="primary",
        disabled=state.processing,
    )

    st.markdown('<p class="sidebar-label">Session</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear chat", use_container_width=True):
            state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("♻️ Reset", use_container_width=True):
            state.result = None
            state.chat_history = []
            st.rerun()

    with st.expander("ℹ️ About this app"):
        st.markdown(
            "- **Transcription:** faster-whisper (local)\n"
            "- **Translation:** Hindi → English\n"
            "- **Summarization:** LangChain + Mistral\n"
            "- **RAG:** ChromaDB + HuggingFace embeddings\n"
        )

# --------------------------------------------------------------------------- #
# Pipeline execution
# --------------------------------------------------------------------------- #

if analyze:
    if not source and not uploaded_file:
        st.warning("Please provide a YouTube URL or upload a file.")
        st.stop()

    temp_path = None
    try:
        state.processing = True

        if uploaded_file:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = temp_path

        progress = st.progress(0, text="Starting analysis...")

        # If run_pipeline supports a progress callback, wire it up here.
        # Otherwise this gives coarse-grained feedback around the call.
        progress.progress(15, text="Downloading / loading source...")
        result = run_pipeline(source, language)
        progress.progress(100, text="Done.")
        progress.empty()

        state.result = result
        st.toast("Analysis complete", icon="✅")

    except Exception as e:
        logger.exception("Pipeline failed")
        st.error(f"Analysis failed: {e}")

    finally:
        state.processing = False
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

if state.result:
    result = state.result

    st.subheader(result.get("title", "Untitled"))
    st.write("")

    transcript = result.get("transcript", "")
    summary = result.get("summary", "")
    actions = result.get("action_items", "")
    decisions = result.get("key_decisions", "")
    questions = result.get("open_questions", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transcript", f"{len(transcript.split()):,} words")
    c2.metric("Summary", f"{len(summary.split()):,} words")
    c3.metric("Action items", len([x for x in actions.splitlines() if x.strip()]))
    c4.metric("Open questions", len([x for x in questions.splitlines() if x.strip()]))

    st.write("")

    tabs = st.tabs(
        ["📄 Summary", "📝 Transcript", "✅ Actions", "🎯 Decisions",
         "❓ Questions", "💬 Chat", "⬇️ Export"]
    )

    with tabs[0]:
        st.markdown(summary or "_No summary generated._")

    with tabs[1]:
        st.text_area("Full transcript", transcript, height=400, label_visibility="collapsed")

    with tabs[2]:
        st.markdown(actions or "_No action items detected._")

    with tabs[3]:
        st.markdown(decisions or "_No decisions detected._")

    with tabs[4]:
        st.markdown(questions or "_No open questions detected._")

    with tabs[5]:
        for role, message in state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        prompt = st.chat_input("Ask anything about this recording...")

        if prompt:
            state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], prompt)
                    except Exception as e:
                        logger.exception("RAG query failed")
                        answer = f"Sorry, I couldn't answer that: {e}"
                st.markdown(answer)

            state.chat_history.append(("assistant", answer))

    with tabs[6]:
        st.caption("Download individual sections or a combined report.")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "📥 Transcript (.txt)", transcript,
                file_name="transcript.txt", mime="text/plain",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "📥 Summary (.txt)", summary,
                file_name="summary.txt", mime="text/plain",
                use_container_width=True,
            )
        with d3:
            combined = (
                f"# {result.get('title', 'Untitled')}\n\n"
                f"## Summary\n{summary}\n\n"
                f"## Action Items\n{actions}\n\n"
                f"## Key Decisions\n{decisions}\n\n"
                f"## Open Questions\n{questions}\n\n"
                f"## Transcript\n{transcript}\n"
            )
            st.download_button(
                "📥 Full report (.md)", combined,
                file_name="meeting_report.md", mime="text/markdown",
                use_container_width=True,
            )
else:
    st.info("Paste a YouTube URL or upload a recording in the sidebar to get started.")

st.divider()
st.caption("faster-whisper · LangChain · ChromaDB · Mistral AI — running locally")