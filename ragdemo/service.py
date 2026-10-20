import re
from collections.abc import Callable
from dataclasses import replace
from time import monotonic
from typing import Protocol

from ragdemo.chunking import TokenBudget, make_chunks
from ragdemo.config import PIPELINE_VERSION, Settings
from ragdemo.errors import BLOCKED_REQUEST, NO_REFERENCES, WITHHELD, DemoError
from ragdemo.ingest import read_pdf, read_text
from ragdemo.models import Answer, Chunk, Conversation, Page, SearchReport
from ragdemo.privacy import PrivacyGuard
from ragdemo.provider import Completion
from ragdemo.retrieval import HybridIndex
from ragdemo.safety import citations_valid, suspicious, unsafe_output


class Provider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def complete(self, question: str, chunks: list[Chunk]) -> Completion: ...


class RagSession:
    def __init__(
        self,
        settings: Settings,
        provider: Provider,
        budget: TokenBudget,
        guard_factory: Callable[[], PrivacyGuard] | None = None,
    ):
        self.settings = settings
        self.pipeline_version = PIPELINE_VERSION
        self.provider = provider
        self.budget = budget
        self._guard_factory = guard_factory
        self.guard: PrivacyGuard | None = None
        self.index: HybridIndex | None = None
        self.conversation = Conversation()
        self.excluded_chunks = 0

    def clear(self) -> None:
        self.index = None
        self.guard = None
        self.conversation = Conversation()
        self.excluded_chunks = 0

    def ingest(self, *, text: str | None = None, pdf: bytes | None = None) -> int:
        # Invalidate first; an exception must never leave a previous document active.
        self.clear()
        try:
            if (text is None) == (pdf is None):
                raise DemoError("Choose either pasted text or one PDF.")
            guard = None
            if self.settings.safety_checks_enabled:
                if self._guard_factory is None:
                    raise DemoError(
                        "Privacy protection is unavailable. Re-initialize the document."
                    )
                guard = self._guard_factory()
            pages = (
                read_text(text, self.settings) if text is not None else read_pdf(pdf, self.settings)
            )
            sanitized = []
            raw_excluded = 0
            for page in pages:
                if guard is None:
                    sanitized.append(page)
                    continue
                # Check raw blocks first: redacting a URL must not hide an exfiltration command.
                blocks = re.split(r"\n\s*\n", page.text)
                safe_blocks = [block for block in blocks if not suspicious(block)]
                raw_excluded += len(blocks) - len(safe_blocks)
                sanitized.append(
                    Page(
                        guard.redact("\n\n".join(safe_blocks)),
                        page.number,
                        tuple(
                            guard.redact(title) for title in page.outline if not suspicious(title)
                        ),
                    )
                )
            chunks, excluded = make_chunks(
                sanitized, self.budget, safety_checks_enabled=self.settings.safety_checks_enabled
            )
            # Defense in depth: every field used for retrieval/generation is inspected.
            if guard is not None and any(
                guard.contains_known_pii(chunk.searchable) for chunk in chunks
            ):
                raise DemoError(WITHHELD)
            vectors = self.provider.embed([chunk.embedding_input for chunk in chunks])
            index = HybridIndex(chunks, vectors)
            self.guard = guard
            self.index = index
            self.excluded_chunks = excluded + raw_excluded
            return len(chunks)
        except DemoError:
            self.clear()
            raise
        except Exception:
            self.clear()
            raise DemoError(
                "Document indexing failed safely. Check the local setup and retry."
            ) from None

    def ask(self, question: str) -> tuple[str, Answer]:
        if self.index is None:
            raise DemoError("Index a document before asking a question.")
        if self.settings.safety_checks_enabled and self.guard is None:
            raise DemoError("Privacy protection is unavailable. Re-index the document.")
        if not question.strip():
            raise DemoError("Enter a question about the document.")
        if len(question) > self.settings.max_question_chars:
            raise DemoError("Keep the question below 4,000 characters.")
        started = monotonic()
        hits = []
        report = SearchReport()
        guard = self.guard.fork() if self.guard else None
        clean_question = guard.redact(question.strip()) if guard else question.strip()
        if self.settings.safety_checks_enabled and (
            suspicious(question) or suspicious(clean_question)
        ):
            answer = Answer(
                BLOCKED_REQUEST,
                status="blocked_question",
                detail="A local override or exfiltration rule matched. The question was not sent to OpenRouter.",
            )
        else:
            query_input = "query: " + clean_question
            if self.budget.count(query_input) >= 512:
                raise DemoError(
                    "The question is too long for the embedding model. Please shorten it."
                )
            vectors = self.provider.embed([query_input])
            report = self.index.search_report(
                clean_question,
                vectors[0],
                self.settings.min_cosine,
                apply_relevance_gate=self.settings.safety_checks_enabled,
            )
            hits = list(report.hits)
            if not hits:
                answer = Answer(
                    NO_REFERENCES,
                    status="no_retrieval",
                    detail="No passage passed the lexical/semantic relevance gate. The chat model was not called.",
                )
            else:
                chunks = [hit.chunk for hit in hits]
                completion = self.provider.complete(clean_question, chunks)
                raw = completion.text
                if not self.settings.safety_checks_enabled:
                    answer = Answer(
                        raw,
                        tuple(chunks),
                        completion.first_token_seconds,
                        monotonic() - started,
                        status="debug",
                        detail="Debug mode: redaction, relevance, and answer checks were bypassed.",
                    )
                elif raw == NO_REFERENCES:
                    answer = Answer(
                        NO_REFERENCES,
                        first_token_seconds=completion.first_token_seconds,
                        safe_answer_seconds=monotonic() - started,
                        status="model_abstained",
                        detail="Passages were retrieved, but the model reported that they do not support an answer.",
                    )
                else:
                    # No raw generated token reaches a UI, history, or logging callback.
                    safe = guard.redact(raw, citation_ids={chunk.id for chunk in chunks})
                    reason = ""
                    if guard.contains_known_pii(safe):
                        reason = "Detected personal data remained after output redaction."
                    elif unsafe_output(raw) or unsafe_output(safe):
                        reason = "The generated answer contained an instruction override, URL, or active markup."
                    elif not citations_valid(safe, {chunk.id for chunk in chunks}):
                        reason = "The generated answer had missing or invalid source citations."
                    if reason:
                        answer = Answer(WITHHELD, status="withheld_output", detail=reason)
                    else:
                        cited = set(re.findall(r"\[(C\d{4,})\]", safe))
                        answer = Answer(
                            safe,
                            tuple(chunk for chunk in chunks if chunk.id in cited),
                            completion.first_token_seconds,
                            monotonic() - started,
                            detail="Answer checked for PII, unsafe content, and citations."
                            + (
                                " Additional PII was redacted from the generated text."
                                if safe != raw
                                else ""
                            ),
                        )
        answer = replace(
            answer,
            retrieval=tuple(hits),
            candidates=report.candidates,
            query_terms=report.query_terms,
            query_corrections=report.corrections,
        )
        self.conversation.messages.append((clean_question, answer))
        return clean_question, answer
