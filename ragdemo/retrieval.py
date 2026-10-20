import re

import faiss
import numpy as np
from rank_bm25 import BM25Plus

from ragdemo.config import TOP_K
from ragdemo.errors import DemoError
from ragdemo.models import Chunk, Hit, SearchReport

# Local, fixed stopwords: no corpus download and no query data sent to third parties.
_STOP = set(
    """a an the and or but if then of to for from in on at by with as is are was were
be been being this that these those it its what which who whom whose how when where why
do does did can could should would will may might please tell me about explain describe
give document reference text say says according information details i you we they their
my your our not no any all there here have has had also into than so such under over section tags
""".split()
)


def terms(text: str) -> list[str]:
    # Keep the anonymous identity, never its original value. Removing placeholders made
    # questions such as "Who is [PERSON_0001]?" impossible to retrieve.
    placeholders = re.findall(r"\[([A-Z_]+_\d{4,})\]", text)
    text = re.sub(r"\[[A-Z_]+_\d{4,}\]", " ", text)
    words = [word for word in re.findall(r"[^\W_]+", text.casefold()) if word not in _STOP]
    return words + ["pii:" + placeholder.casefold() for placeholder in placeholders]


def normalized(vectors: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 1024 or not np.isfinite(matrix).all():
        raise DemoError("The embedding provider returned invalid or incompatible vectors.")
    if np.any(np.linalg.norm(matrix, axis=1) < 1e-10):
        raise DemoError("The embedding provider returned an empty vector.")
    matrix = np.ascontiguousarray(matrix.copy())
    faiss.normalize_L2(matrix)
    return matrix


def _one_edit_away(first: str, second: str) -> bool:
    """One insertion, deletion, substitution, or adjacent transposition."""
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        positions = [i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b]
        if len(positions) == 1:
            return True
        return (
            len(positions) == 2
            and positions[1] == positions[0] + 1
            and first[positions[0]] == second[positions[1]]
            and first[positions[1]] == second[positions[0]]
        )
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1
    return shorter[index:] == longer[index + 1 :]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]):
        matrix = normalized(embeddings)
        if len(chunks) != matrix.shape[0] or not chunks:
            raise DemoError("The document index could not be built completely.")
        self.chunks = chunks
        self._tokens = [terms(chunk.searchable) for chunk in chunks]
        self._vocabulary = {term for row in self._tokens for term in row}
        self._bm25 = BM25Plus([tokens or ["__empty__"] for tokens in self._tokens])
        self._vectors = matrix
        self._faiss = faiss.IndexFlatIP(matrix.shape[1])
        self._faiss.add(matrix)

    def search(
        self,
        question: str,
        embedding: list[float],
        min_cosine: float,
        *,
        apply_relevance_gate: bool = True,
    ) -> list[Hit]:
        return list(
            self.search_report(
                question, embedding, min_cosine, apply_relevance_gate=apply_relevance_gate
            ).hits
        )

    def search_report(
        self,
        question: str,
        embedding: list[float],
        min_cosine: float,
        *,
        apply_relevance_gate: bool = True,
    ) -> SearchReport:
        query_terms = set(terms(question))
        if not query_terms and apply_relevance_gate:
            return SearchReport()
        original_terms = tuple(sorted(query_terms))
        corrections = []
        for term in sorted(query_terms):
            # Never fuzzy-match numbers, short words, or anonymous/private identity tokens.
            if term in self._vocabulary or len(term) < 6 or not term.isalpha():
                continue
            options = [
                word
                for word in self._vocabulary
                if word.isalpha() and len(word) >= 6 and _one_edit_away(term, word)
            ]
            if len(options) == 1:
                query_terms.remove(term)
                query_terms.add(options[0])
                corrections.append((term, options[0]))
        query = normalized([embedding])
        count = min(10, len(self.chunks))
        _, vector_ids = self._faiss.search(query, count)
        raw_cosines = (self._vectors @ query[0]).tolist()
        lexical_scores = self._bm25.get_scores(list(query_terms))
        lexical_ids = sorted(
            (
                index
                for index in range(len(self.chunks))
                if query_terms.intersection(self._tokens[index])
            ),
            key=lambda index: (-lexical_scores[index], index),
        )[:count]
        fusion: dict[int, float] = {}
        for ranking in (vector_ids[0].tolist(), lexical_ids):
            for rank, index in enumerate(ranking, 1):
                if index >= 0:
                    fusion[index] = fusion.get(index, 0.0) + 1 / (60 + rank)
        results = []
        candidates = []
        seen = set()
        for index in sorted(fusion, key=lambda key: (-fusion[key], key)):
            matches = len(query_terms.intersection(self._tokens[index]))
            cosine = raw_cosines[index]
            # Anonymous-identity-only questions require the same identity in the source.
            # Ordinary single-word questions may match semantically (e.g. singular/plural).
            identity_only = all(term.startswith("pii:") for term in query_terms)
            relevant = (
                not apply_relevance_gate
                or matches >= min(2, len(query_terms))
                or (not identity_only and cosine >= min_cosine)
            )
            chunk = self.chunks[index]
            matched_terms = tuple(sorted(query_terms.intersection(self._tokens[index])))
            if not apply_relevance_gate:
                reason = "Debug: relevance gate bypassed"
            elif matches >= min(2, len(query_terms)):
                reason = "Lexical match (after any displayed spelling correction)"
            elif relevant:
                reason = f"Semantic match: cosine >= {min_cosine:.2f}"
            elif identity_only:
                reason = "Rejected: anonymous identity not present in this chunk"
            else:
                reason = f"Rejected: insufficient lexical overlap and cosine < {min_cosine:.2f}"
            hit = Hit(chunk, cosine, matches, fusion[index], matched_terms, relevant, reason)
            candidates.append(hit)
            signature = re.sub(r"\s+", " ", chunk.text).strip().casefold()
            if relevant and signature not in seen and len(results) < TOP_K:
                seen.add(signature)
                results.append(hit)
        return SearchReport(tuple(results), tuple(candidates), original_terms, tuple(corrections))
