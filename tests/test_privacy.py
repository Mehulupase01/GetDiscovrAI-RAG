import pytest

from ragdemo.errors import DemoError
from ragdemo.privacy import PrivacyGuard


@pytest.mark.integration
def test_broad_pii_redaction_and_stable_placeholders(analyzer):
    guard = PrivacyGuard(analyzer)
    source = (
        "Alice Morgan lives at 123 Maple Street, Boston. "
        "Email alice.morgan@example.com or alice [at] example [dot] com. "
        "Call +1 202-555-0147. Her SSN is 123-45-6789. "
        "Credit card: 4111 1111 1111 1111. IBAN: DE89370400440532013000."
    )
    clean = guard.redact(source)
    for value in (
        "Alice Morgan",
        "123 Maple Street",
        "Boston",
        "alice.morgan@example.com",
        "alice [at] example [dot] com",
        "+1 202-555-0147",
        "123-45-6789",
        "4111 1111 1111 1111",
        "DE89370400440532013000",
    ):
        assert value.casefold() not in clean.casefold()
    assert not guard.contains_known_pii(clean)
    assert guard.redact(source) == clean
    assert guard.redact(clean) == clean
    assert guard.contains_known_pii("ALICE.MORGAN@example.com")
    assert guard.contains_known_pii("alice . morgan @ example . com")


@pytest.mark.integration
def test_preserve_policy_numbers_and_benign_content(analyzer):
    clean = PrivacyGuard(analyzer).redact(
        "Refund requests are accepted within 30 days. Shipping takes 5 business days."
    )
    assert "30 days" in clean
    assert "5 business days" in clean


def test_privacy_failure_is_closed():
    class BrokenAnalyzer:
        def analyze(self, **kwargs):
            raise RuntimeError("sensitive raw text should not be exposed")

    with pytest.raises(DemoError, match="Privacy inspection failed") as error:
        PrivacyGuard(BrokenAnalyzer()).redact("alice@example.com")
    assert "sensitive" not in str(error.value)


def test_only_trusted_placeholders_and_supplied_citations_are_protected(analyzer):
    guard = PrivacyGuard(analyzer)
    guard.redact("Her SSN is 123-45-6789.")
    # An unrecognized obfuscation of an already-detected value must stop processing.
    with pytest.raises(DemoError, match="Privacy inspection failed"):
        guard.redact("Her SSN is [US_SSN_123456789].")
    assert guard.contains_known_pii("[US_SSN_123456789]")
    assert "[C0001]" in guard.redact(
        "The refund deadline is 30 days. [C0001]", citation_ids={"C0001"}
    )


def test_birth_dates_and_ip_addresses(analyzer):
    clean = PrivacyGuard(analyzer).redact("DOB: 1990-05-12. IP address: 192.168.1.10.")
    assert "1990-05-12" not in clean
    assert "192.168.1.10" not in clean


def test_birth_date_value_cannot_reappear_without_its_original_label(analyzer):
    guard = PrivacyGuard(analyzer)
    clean = guard.redact("DOB: 1990-05-12.")
    assert "DOB:" in clean
    assert "1990-05-12" not in clean
    assert "1990-05-12" not in guard.redact("The date is 1990-05-12.")


def test_names_keep_identity_across_case_and_whitespace(analyzer):
    guard = PrivacyGuard(analyzer)
    source = guard.redact("Alice   Morgan is the refund manager.")
    query = guard.redact("Who is alice morgan?")
    assert "[PERSON_0001]" in source
    assert "[PERSON_0001]" in query
    assert "alice" not in query.casefold()
