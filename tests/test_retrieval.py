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


@pytest.mark.parametrize(
    "question", ["what is velnord?", "what is velonrod?", "what is velonordd?"]
)
def test_unique_brand_typo_matches_lexically_even_with_low_cosine(question):
    index = HybridIndex(
        [
            Chunk("C0001", "VeloNord is a fictional e-bike company with 800 employees.", "Company"),
        ],
        [vector(0)],
    )
    report = index.search_report(question, vector(1), 0.3)
    assert len(report.hits) == 1
    assert report.hits[0].cosine == 0
    assert report.hits[0].lexical_matches == 1
    assert report.hits[0].matched_terms == ("velonord",)
    assert report.corrections
    assert report.hits[0].eligible


def test_ambiguous_spelling_correction_is_not_applied():
    index = HybridIndex(
        [
            Chunk("C0001", "Velonord and Velinord are product names.", "Products"),
        ],
        [vector(0)],
    )
    report = index.search_report("velanord", vector(1), 0.3)
    assert not report.hits
    assert not report.corrections


@pytest.mark.parametrize("question", ["[PERSON_0002]", "801", "tim"])
def test_spelling_never_matches_private_ids_numbers_or_short_names(question):
    index = HybridIndex(
        [
            Chunk("C0001", "[PERSON_0001] and Tom manage 800 employees.", "Contacts"),
        ],
        [vector(0)],
    )
    report = index.search_report(question, vector(1), 0.3)
    assert not report.hits
    assert not report.corrections


def test_rejected_candidates_explain_zero_matching_terms_and_low_cosine():
    index = HybridIndex([Chunk("C0001", "VeloNord makes e-bikes.", "Company")], [vector(0)])
    report = index.search_report("volcanic eruptions", vector(1), 0.3)
    assert not report.hits
    assert report.query_terms == ("eruptions", "volcanic")
    assert len(report.candidates) == 1
    assert not report.candidates[0].eligible
    assert report.candidates[0].lexical_matches == 0
    assert not report.candidates[0].matched_terms
    assert "cosine < 0.30" in report.candidates[0].reason


def test_definition_is_not_displaced_by_short_incidental_company_mentions():
    chunks = [
        Chunk(
            "C0001",
            "VeloNord (a fictional company) makes e-bikes. It has about 800 employees. "
            + "Dealers distribute the products across several markets. " * 10,
            "The scenario",
        ),
        Chunk("C0002", "VeloNord uses Microsoft 365.", "Background"),
        Chunk("C0003", "Ask VeloNord five questions.", "What to bring"),
        Chunk("C0004", "Build a system for VeloNord.", "Your task"),
    ]
    index = HybridIndex(chunks, [vector(0)] * len(chunks))
    report = index.search_report("what is Velnord ?", vector(1), 0.3)
    assert len(report.hits) == 3
    assert report.hits[0].chunk.id == "C0001"
    assert report.hits[0].definition_match
    assert "Direct definition" in report.hits[0].reason


@pytest.mark.parametrize("vectors", [[[1, 0]], [[float("nan")] * 1024], [[0.0] * 1024]])
def test_invalid_embeddings_rejected(vectors):
    with pytest.raises(DemoError):
        HybridIndex([Chunk("C0001", "Refunds", "Policy")], vectors)
