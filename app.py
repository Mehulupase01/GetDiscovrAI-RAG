from dataclasses import replace

import streamlit as st

from ragdemo.chunking import TokenBudget
from ragdemo.config import PIPELINE_VERSION, TOP_K, Settings
from ragdemo.errors import DemoError
from ragdemo.privacy import PrivacyGuard, build_analyzer
from ragdemo.provider import OpenRouterProvider
from ragdemo.service import RagSession

st.set_page_config(page_title="GetDiscovrAI · Local RAG", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def local_analyzer():
    return build_analyzer()


@st.cache_resource(show_spinner=False)
def local_budget():
    return TokenBudget.load()


def initialize(safety_checks_enabled: bool) -> RagSession:
    settings = replace(Settings.load(), safety_checks_enabled=safety_checks_enabled)
    settings.require_key()
    budget = local_budget()
    analyzer = local_analyzer() if safety_checks_enabled else None
    return RagSession(
        settings,
        OpenRouterProvider(settings),
        budget,
        (lambda: PrivacyGuard(analyzer)) if analyzer is not None else None,
    )


def show_answer(answer) -> None:
    st.text(answer.text)  # Never render untrusted Markdown, HTML, images, or clickable links.
    for number, chunk in enumerate(answer.sources, 1):
        with st.expander(f"Source {number} [{chunk.id}]"):
            st.text(chunk.section)
            if chunk.pages:
                st.caption("PDF pages: " + ", ".join(map(str, chunk.pages)))
            st.text(chunk.text)
    if answer.detail:
        with st.expander("Answer details"):
            st.text(answer.detail)
            if answer.retrieval:
                st.table(
                    [
                        {
                            "Chunk": hit.chunk.id,
                            "Cosine similarity": round(hit.cosine, 3),
                            "Matching terms": hit.lexical_matches,
                            "Fusion score": round(hit.fusion_score, 4),
                        }
                        for hit in answer.retrieval
                    ]
                )
    if answer.safe_answer_seconds is not None:
        st.caption(
            f"Response time: {answer.safe_answer_seconds:.2f}s · "
            f"Provider first token: {answer.first_token_seconds:.2f}s"
            if answer.first_token_seconds is not None
            else f"Response time: {answer.safe_answer_seconds:.2f}s"
        )


st.title("GetDiscovrAI")
st.write("Ask questions about one reference document.")
st.caption(f"Hybrid retrieval · Top K = {TOP_K}")

try:
    default_settings = Settings.load()
except DemoError as exc:
    st.error(str(exc))
    st.stop()

safety_enabled = st.toggle("Enable safety checks", value=default_settings.safety_checks_enabled)
if not safety_enabled:
    st.info(
        "Debug mode: redaction, injection screening, relevance filtering, and answer checks are off. "
        "Document text and questions are sent to OpenRouter without redaction."
    )

previous_rag = st.session_state.get("rag")
safety_changed = previous_rag is not None and (
    getattr(previous_rag.settings, "safety_checks_enabled", True) != safety_enabled
    or getattr(previous_rag, "pipeline_version", None) != PIPELINE_VERSION
)
if safety_changed:
    previous_rag.clear()
    st.session_state.pop("rag", None)

if st.button("Clear document and chat"):
    if "rag" in st.session_state:
        st.session_state.rag.clear()
    st.session_state.pop("rag", None)
    st.session_state.pop("reference_text", None)
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1
    st.rerun()

with st.container(border=True):
    mode = st.radio("Reference source", ["Paste text", "Upload PDF"], horizontal=True)
    text = None
    upload = None
    if mode == "Paste text":
        text = st.text_area("Reference text", height=220, key="reference_text", max_chars=200_000)
    else:
        upload = st.file_uploader(
            "Text PDF (up to 10 MB / 100 pages)",
            type=["pdf"],
            key=f"pdf_{st.session_state.get('upload_version', 0)}",
        )
    index_requested = st.button("Index document", type="primary")
    # Rebuild existing input automatically when a setting changes; never reuse a redacted index.
    if index_requested or (safety_changed and (text or upload is not None)):
        # Clear old state even when the new upload/input or asset initialization fails.
        if "rag" in st.session_state:
            st.session_state.rag.clear()
        try:
            if mode == "Upload PDF" and upload is None:
                raise DemoError("Choose a PDF before indexing.")
            with st.status(
                "Preparing local privacy protection and indexing…"
                if safety_enabled
                else "Indexing document…",
                expanded=False,
            ) as status:
                rag = initialize(safety_enabled)
                st.session_state.rag = rag
                count = (
                    rag.ingest(text=text)
                    if mode == "Paste text"
                    else rag.ingest(pdf=upload.getvalue())
                )
                status.update(label=f"Ready · {count} chunks indexed", state="complete")
        except DemoError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The document could not be indexed safely. Check the local setup and retry.")

rag = st.session_state.get("rag")
ready = rag is not None and rag.index is not None
if ready:
    summary = f"Document ready · {len(rag.index.chunks)} chunks"
    if safety_enabled:
        summary += f" · {rag.excluded_chunks} unsafe blocks excluded"
    st.success(summary)
    preview = rag.index.chunks[:5]
    with st.expander(f"First {len(preview)} indexed chunks", expanded=True):
        for chunk in preview:
            st.caption(f"[{chunk.id}] · {chunk.section}")
            if chunk.pages:
                st.caption("PDF pages: " + ", ".join(map(str, chunk.pages)))
            st.text(chunk.text)
    for question, answer in rag.conversation.messages:
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            show_answer(answer)
else:
    st.info("Index a document to begin.")

question = st.chat_input("Ask an independent question about the reference", disabled=not ready)
if question and ready:
    try:
        with st.status(
            "Retrieving evidence, generating, and inspecting the complete answer…"
            if safety_enabled
            else "Retrieving evidence and generating…",
            expanded=False,
        ) as status:
            clean_question, answer = rag.ask(question)
            status.update(
                label="Answer inspection complete" if safety_enabled else "Answer ready",
                state="complete",
            )
        with st.chat_message("user"):
            st.text(clean_question)
        with st.chat_message("assistant"):
            show_answer(answer)
    except DemoError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The request failed safely. No unchecked answer was displayed.")

if safety_enabled:
    st.caption(
        "Reference data is kept in local session memory. Detected PII is redacted locally before "
        "OpenRouter calls. Automated detection and injection checks can miss cases; use synthetic "
        "data for this demonstration. Scanned PDFs and conversational follow-ups are unsupported."
    )
else:
    st.caption(
        "Safety checks are disabled. Reference data stays in local session memory and is sent to OpenRouter without redaction."
    )
