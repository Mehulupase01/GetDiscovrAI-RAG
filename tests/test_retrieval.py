import numpy as np
import pytest

from ragdemo.errors import DemoError
from ragdemo.models import Chunk
from ragdemo.retrieval import HybridIndex


def vector(axis):
    value = np.zeros(1024)
    value[axis] = 1
    return value.tolist()


def test_single_chunk_positive_bm25_and_exact_term_gate():
    index = HybridIndex([Chunk("C0001", "Refund requests within 30 days.", "Policy")], [vector(0)])
    hit = index.search("refund", vector(1), 0.5)
    assert len(hit) == 1
    assert hit[0].lexical_matches == 1
    assert hit[0].fusion_score == pytest.approx(2 / 61)
    assert index.search("volcano", vector(1), 0.5) == []
    assert index.search("refunds", vector(0), 0.5)  # Single-term semantic match.


def test_semantic_paraphrase_and_no_relevant_evidence():
    index = HybridIndex([Chunk("C0001", "Refund requests within 30 days.", "Policy")], [vector(0)])
    assert index.search("get money back", vector(0), 0.5)
    assert index.search("volcano eruption", vector(1), 0.5) == []


def test_top_three_limit_and_deduplication():
    texts = [
        "Refund requests within 30 days.",
        "Refund requests within 30 days.",
        "Refund requests need receipts.",
        "Refund requests exclude software.",
        "Refund requests processed weekly.",
    ]
    chunks = [Chunk(f"C{i + 1:04d}", text, "Refunds") for i, text in enumerate(texts)]
    index = HybridIndex(chunks, [vector(0)] * len(chunks))
    hits = index.search("refund requests", vector(0), 0.5)
    assert len(hits) == 3
    assert len({hit.chunk.text for hit in hits}) == 3


def test_redacted_identity_is_searchable_without_exposing_value():
    chunks = [
        Chunk("C0001", "[PERSON_0001] manages refunds.", "Contacts"),
        Chunk("C0002", "[PERSON_0002] manages shipping.", "Contacts"),
    ]
    index = HybridIndex(chunks, [vector(0), vector(1)])
    assert index.search("Who is [PERSON_0001]?", vector(1), 0.5)[0].chunk.id == "C0001"
    assert index.search("Who is [PERSON_0003]?", vector(0), 0.5) == []


@pytest.mark.parametrize("vectors", [[[1, 0]], [[float("nan")] * 1024], [[0.0] * 1024]])
def test_invalid_embeddings_rejected(vectors):
    with pytest.raises(DemoError):
        HybridIndex([Chunk("C0001", "Refunds", "Policy")], vectors)
