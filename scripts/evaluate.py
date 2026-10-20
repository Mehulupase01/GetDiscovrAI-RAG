"""Evaluate the live relevance gate using only the shipped synthetic reference."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ragdemo.chunking import TokenBudget
from ragdemo.config import Settings
from ragdemo.errors import DemoError
from ragdemo.privacy import PrivacyGuard, build_analyzer
from ragdemo.provider import OpenRouterProvider
from ragdemo.retrieval import normalized, terms
from ragdemo.service import RagSession

CASES = (
    ("What is Velnord?", "Velnord"),
    ("Velnord", "Velnord"),
    ("Who is Alice Morgan?", "Contacts"),
    ("Who is Bob Stone?", None),
    ("refunds", "Refunds"),
    ("What is the refund deadline?", "Refunds"),
    ("How long do I have to get my money back?", "Refunds"),
    ("What are the refund exceptions?", "Exceptions"),
    ("How long does express shipping take?", "Delivery"),
    ("Does the warranty cover accidental damage?", "Warranty"),
    ("What is the contact email address?", "Contacts"),
    ("Explain volcanic eruptions.", None),
    ("Who invented the airplane?", None),
    ("Explain ocean currents.", None),
    ("How do I bake a chocolate cake?", None),
    ("What is the formula for nuclear fusion?", None),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save synthetic evaluation results as JSON")
    args = parser.parse_args()
    try:
        settings = replace(Settings.load(), safety_checks_enabled=True)
        settings.require_key()
        analyzer = build_analyzer()
        provider = OpenRouterProvider(settings)
        session = RagSession(settings, provider, TokenBudget.load(), lambda: PrivacyGuard(analyzer))
        session.ingest(text=Path("examples/reference.md").read_text(encoding="utf-8"))
        questions = [session.guard.redact(question) for question, _ in CASES]
        vectors = provider.embed(["query: " + question for question in questions])
        results = []
        for (original, section), question, vector in zip(CASES, questions, vectors, strict=True):
            hits = session.index.search(question, vector, settings.min_cosine)
            scores = session.index._vectors @ normalized([vector])[0]
            passed = (
                any(section in hit.chunk.section for hit in hits) if section else not hits
            ) and len(hits) <= 3
            results.append(
                {
                    "question": original,
                    "expected_section": section,
                    "passed": passed,
                    "best_cosine": round(float(max(scores)), 4),
                    "max_lexical_overlap": max(
                        len(set(terms(question)).intersection(terms(chunk.searchable)))
                        for chunk in session.index.chunks
                    ),
                    "selected_sections": [hit.chunk.section for hit in hits],
                }
            )
        report = {
            "safety_checks_enabled": True,
            "top_k": 3,
            "threshold": settings.min_cosine,
            "passed": sum(row["passed"] for row in results),
            "total": len(results),
            "results": results,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if all(row["passed"] for row in results) else 1
    except DemoError as exc:
        print(str(exc))
        return 1
    except Exception:
        print("Evaluation failed safely. No raw document or provider error body was logged.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
