from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    text: str
    number: int | None = None
    outline: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    section: str
    pages: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def searchable(self) -> str:
        return f"Section: {self.section}\nTags: {', '.join(self.tags)}\n{self.text}"

    @property
    def embedding_input(self) -> str:
        return "document: " + self.searchable


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    cosine: float
    lexical_matches: int
    fusion_score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[Chunk, ...] = ()
    first_token_seconds: float | None = None
    safe_answer_seconds: float | None = None
    status: str = "answered"
    detail: str = ""
    retrieval: tuple[Hit, ...] = ()


@dataclass
class Conversation:
    """Only inspected text belongs here. No provider history is retained."""

    messages: list[tuple[str, Answer]] = field(default_factory=list)
