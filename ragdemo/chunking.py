import re
from dataclasses import dataclass

from tokenizers import Tokenizer

from ragdemo.config import TOKENIZER_MODEL, TOKENIZER_REVISION
from ragdemo.errors import DemoError
from ragdemo.models import Chunk, Page
from ragdemo.safety import suspicious


class TokenBudget:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def load(cls) -> "TokenBudget":
        try:
            # Rust tokenizer JSON only; no model weights and no remote Python code.
            tokenizer = Tokenizer.from_pretrained(TOKENIZER_MODEL, revision=TOKENIZER_REVISION)
            tokenizer.no_truncation()
            tokenizer.no_padding()
            return cls(tokenizer)
        except Exception:
            raise DemoError(
                "The embedding tokenizer could not load. Check connectivity and retry."
            ) from None

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def prefix(self, text: str, tokens: int) -> str:
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= tokens:
            return text
        return text[: encoding.offsets[tokens - 1][1]]

    def tail(self, text: str, tokens: int) -> str:
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= tokens:
            return text
        return text[encoding.offsets[-tokens][0] :]


@dataclass(frozen=True)
class _Block:
    text: str
    section: str
    page: int | None
    tags: tuple[str, ...]


def _heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
    if match:
        return len(match[1]), match[2]
    match = re.match(r"^(\d+(?:\.\d+)*)(?:\.|\))?\s+([A-Z][^.!?]{0,100})$", line)
    if match and len(line.split()) <= 14:
        return min(match[1].count(".") + 1, 6), match[2]
    return None


def _key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _blocks(pages: list[Page], safety_checks_enabled: bool = True) -> list[_Block]:
    # TOC lines are metadata candidates only. Confirm against actual body headings.
    candidates: dict[str, str] = {}
    for page in pages:
        for line in page.text.splitlines():
            match = re.match(
                r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(.+?)\s*(?:\.{2,}|\s{3,})\s*\d+\s*$", line
            )
            if match:
                candidates[_key(match[1])] = match[1].strip()
    headings: list[str] = []
    result: list[_Block] = []
    paragraph: list[str] = []
    page_number = None

    def flush() -> None:
        if paragraph:
            section = " > ".join(headings) or "Document"
            tags = tuple(candidates[_key(h)] for h in headings if _key(h) in candidates)
            result.append(_Block("\n".join(paragraph), section, page_number, tags))
            paragraph.clear()

    for page in pages:
        page_number = page.number
        outline = [
            title for title in page.outline if not safety_checks_enabled or not suspicious(title)
        ]
        if outline:
            headings = [outline[-1]]

        for line in page.text.splitlines():
            line = line.strip()
            if not line:
                flush()
                continue
            # TOC/index rows cannot serve as substantive evidence.
            if re.search(r"\.{2,}\s*\d+\s*$", line):
                flush()
                continue
            heading = _heading(line)
            if not heading and line in outline:
                heading = (1, line)
            if heading:
                flush()
                level, title = heading
                headings = headings[: level - 1] + [title]
            else:
                paragraph.append(line)
        flush()
    return result


def make_chunks(
    pages: list[Page], budget: TokenBudget, *, safety_checks_enabled: bool = True
) -> tuple[list[Chunk], int]:
    chunks: list[Chunk] = []
    excluded = 0
    current = ""
    section = ""
    section_key = ""
    origins: list[tuple[int, int, int]] = []
    tags: tuple[str, ...] = ()

    def candidate(text: str) -> Chunk:
        pages = tuple(sorted({page for _, _, page in origins}))
        return Chunk(f"C{len(chunks) + 1:04d}", text, section, pages, tags)

    def append(text: str, page: int | None) -> None:
        nonlocal current
        text = text.strip()
        start = len(current) + (2 if current else 0)
        current = (current + "\n\n" + text).strip()
        if page is not None and text:
            origins.append((start, len(current), page))

    def overlap() -> tuple[str, list[tuple[int, int, int]]]:
        tail = budget.tail(current, 48).lstrip()
        offset = len(current) - len(tail)
        ranges = [
            (max(start, offset) - offset, end - offset, page)
            for start, end, page in origins
            if end > offset
        ]
        return tail, ranges

    def flush() -> None:
        nonlocal current, origins, excluded
        if current.strip():
            chunk = candidate(current.strip())
            if safety_checks_enabled and suspicious(chunk.searchable):
                excluded += 1
            else:
                chunks.append(chunk)
        current = ""
        origins = []

    for block in _blocks(pages, safety_checks_enabled):
        if safety_checks_enabled and (suspicious(block.text) or suspicious(block.section)):
            flush()
            excluded += 1
            continue
        if block.section != section_key:
            flush()
            section_key = block.section
            section = budget.prefix(block.section, 64)
            tags = tuple(budget.prefix(tag, 24) for tag in block.tags[:3])
        # Paragraphs stay intact if possible; sentence boundaries split large paragraphs.
        pieces = [block.text]
        if budget.count(candidate(block.text).embedding_input) > 320:
            pieces = re.split(r"(?<=[.!?])\s+", block.text)
        for piece in pieces:
            piece = piece.strip()
            while piece:
                joined = (current + "\n\n" + piece).strip()
                if budget.count(candidate(joined).embedding_input) <= 320:
                    append(piece, block.page)
                    break
                if current:
                    tail, tail_origins = overlap()
                    flush()
                    current, origins = tail, tail_origins
                    joined = (current + "\n\n" + piece).strip()
                    if budget.count(candidate(joined).embedding_input) <= 448:
                        append(piece, block.page)
                        break
                    # Avoid creating an overlap-only chunk before a long sentence.
                # Use text offsets so Unicode and placeholders are not decoded incorrectly.
                available = 448 - budget.count(candidate(current + "\n\n").embedding_input)
                prefix = budget.prefix(piece, max(1, available))
                while (
                    prefix
                    and budget.count(candidate((current + "\n\n" + prefix).strip()).embedding_input)
                    > 448
                ):
                    prefix = prefix[:-1]
                if not prefix:
                    raise DemoError("A document section could not fit the embedding token budget.")
                append(prefix, block.page)
                piece = piece[len(prefix) :].lstrip()
                tail, tail_origins = overlap()
                flush()
                if piece:
                    current, origins = tail, tail_origins
    flush()
    if not chunks:
        raise DemoError("No usable reference text remained after document safety checks.")
    if any(budget.count(chunk.embedding_input) >= 512 for chunk in chunks):
        raise DemoError("A chunk exceeded the embedding model's input limit.")
    return chunks, excluded
