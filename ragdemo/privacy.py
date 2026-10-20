import re
from collections import defaultdict

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ragdemo.errors import DemoError
from ragdemo.safety import canonical

_PLACEHOLDER = re.compile(r"\[(?:[A-Z_]+_\d{4,}|C\d{4,})\]")


def build_analyzer() -> AnalyzerEngine:
    """Load local assets only; missing NLP support must never weaken the policy."""
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
        )
        # Fail before Presidio's engine has a chance to auto-download a missing model.
        import spacy.util

        if not spacy.util.is_package("en_core_web_lg"):
            raise RuntimeError("Missing local model")
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        extra = (
            (
                "EMAIL_ADDRESS",
                "obfuscated_email",
                r"\b[A-Za-z0-9._%+-]+\s*(?:\[at\]|\(at\)|\s+at\s+)\s*[A-Za-z0-9.-]+\s*(?:\[dot\]|\(dot\)|\s+dot\s+)\s*[A-Za-z]{2,}\b",
            ),
            (
                "STREET_ADDRESS",
                "street_address",
                r"\b\d{1,6}\s+(?:[A-Za-z][A-Za-z.'-]*\s+){1,6}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)(?:\.|\b)(?:\s*,?\s*(?:Apt|Unit|Suite|#)\s*[A-Za-z0-9-]+)?",
            ),
            ("PHONE_NUMBER", "international_phone", r"(?<!\w)\+\d[\d ().-]{7,}\d\b"),
            (
                "SECRET",
                "api_credential",
                r"\b(?:sk-or-v1-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{20,})\b",
            ),
            ("US_SSN", "labelled_ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
            (
                "DATE_OF_BIRTH",
                "labelled_birth_date",
                r"(?i)\b(?:date of birth|DOB|born on)\s*[:=-]?\s*(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b",
            ),
        )
        for entity, name, regex in extra:
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=entity,
                    patterns=[Pattern(name, regex, 0.85)],
                    supported_language="en",
                )
            )
        return analyzer
    except Exception:
        raise DemoError(
            "Privacy protection could not initialize. Run uv sync and restart."
        ) from None


class PrivacyGuard:
    """Per-document state. Original detected values never leave this object."""

    def __init__(self, analyzer: AnalyzerEngine):
        self._analyzer = analyzer
        self._anonymizer = AnonymizerEngine()
        self._known: dict[str, tuple[str, str]] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self.redaction_count = 0

    def _placeholder(self, entity: str, value: str) -> str:
        key = re.sub(r"\s+", " ", canonical(value)).casefold().strip()
        if key not in self._known:
            self._counts[entity] += 1
            self._known[key] = (entity, f"[{entity}_{self._counts[entity]:04d}]")
        return self._known[key][1]

    def redact(self, text: str, *, citation_ids: set[str] | None = None) -> str:
        try:
            text = canonical(text)
            trusted = {placeholder for _, placeholder in self._known.values()}
            trusted.update(f"[{value}]" for value in (citation_ids or set()))
            protected = [
                (m.start(), m.end()) for m in _PLACEHOLDER.finditer(text) if m.group() in trusted
            ]
            results = [
                item
                for item in self._analyzer.analyze(text=text, language="en", score_threshold=0.35)
                # Organizations, demographic labels, and arbitrary dates are not always PII.
                # Birth dates have a separate contextual recognizer.
                if item.entity_type not in {"DATE_TIME", "ORGANIZATION", "NRP", "AGE"}
                and not (
                    item.entity_type == "PERSON"
                    and text[item.start : item.end].casefold().strip()
                    in {"email", "phone", "contact", "fax", "ssn", "iban", "dob", "address"}
                )
            ]
            # The DOB recognizer uses the label as context, but only the date is PII.
            # Remember the value itself so it is also caught without its label in output.
            for item in results:
                if item.entity_type == "DATE_OF_BIRTH":
                    label = re.match(
                        r"(?:date of birth|DOB|born on)\s*[:=-]?\s*",
                        text[item.start : item.end],
                        re.I,
                    )
                    if label:
                        item.start += label.end()
            # Recognize already-seen values even if the NER model misses them in a new context.
            for value, (entity, _) in list(self._known.items()):
                pattern = re.escape(value).replace(r"\ ", r"\s+")
                for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", text, re.I):
                    results.append(RecognizerResult(entity, match.start(), match.end(), 1.0))
            results = [
                item
                for item in results
                if not any(item.start < end and item.end > start for start, end in protected)
            ]
            # Merge overlapping spans before assigning stable placeholders.
            merged: list[RecognizerResult] = []
            for item in sorted(results, key=lambda x: (x.start, -x.end, -x.score)):
                if merged and item.start < merged[-1].end:
                    prev = merged[-1]
                    if item.end > prev.end:
                        merged[-1] = RecognizerResult(
                            prev.entity_type, prev.start, item.end, max(prev.score, item.score)
                        )
                else:
                    merged.append(item)
            operators = {}
            for item in merged:
                self._placeholder(item.entity_type, text[item.start : item.end])
                operators[item.entity_type] = OperatorConfig(
                    "custom",
                    {
                        "lambda": lambda value, entity=item.entity_type: self._placeholder(
                            entity, value
                        )
                    },
                )
            self.redaction_count += len(merged)
            sanitized = self._anonymizer.anonymize(
                text=text, analyzer_results=merged, operators=operators
            ).text
            if self.contains_known_pii(sanitized):
                raise RuntimeError("Residual detected value")
            return sanitized
        except Exception:
            raise DemoError(
                "Privacy inspection failed. No unchecked content was sent or displayed."
            ) from None

    def contains_known_pii(self, text: str) -> bool:
        trusted = {placeholder for _, placeholder in self._known.values()}
        text = _PLACEHOLDER.sub(
            lambda match: "" if match.group() in trusted else match.group(), canonical(text)
        ).casefold()
        compact = re.sub(r"[\W_]", "", text)
        for value in self._known:
            if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I):
                return True
            needle = re.sub(r"[\W_]", "", value)
            if len(needle) >= 6 and needle in compact:
                return True
        return False
