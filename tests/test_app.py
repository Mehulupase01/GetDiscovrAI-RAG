from pathlib import Path

from streamlit.testing.v1 import AppTest

from ragdemo.config import Settings


def test_dashboard_index_chat_replace_failure_and_clear(monkeypatch, analyzer, budget, provider):
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: Settings(api_key="test-only")))
    monkeypatch.setattr("ragdemo.privacy.build_analyzer", lambda: analyzer)
    monkeypatch.setattr("ragdemo.chunking.TokenBudget.load", classmethod(lambda cls: budget))
    monkeypatch.setattr("ragdemo.provider.OpenRouterProvider", lambda settings: provider)
    app = AppTest.from_file(str(Path("app.py").resolve())).run(timeout=30)
    assert not app.exception
    assert app.chat_input[0].disabled
    app.text_area[0].set_value(
        "# Refunds\n\nRefund requests are accepted within 30 days. Email alice@example.com."
    )
    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("What is the refund deadline?").run(timeout=30)
    assert not app.exception
    assert any("30 days" in item.value for item in app.text)
    assert all("alice@example.com" not in item.value for item in app.text)
    assert any("Top K = 3" in item.value for item in app.caption)
    assert any(item.label == "Answer details" for item in app.expander)
    assert any("Answer checked" in item.value for item in app.text)
    app.text_area[0].set_value("")
    app.button[1].click().run(timeout=30)
    assert app.chat_input[0].disabled
    assert app.error
    assert not app.chat_message
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.text_area[0].value == ""


def test_dashboard_explains_spelling_matches_and_rejected_candidates(
    monkeypatch, analyzer, budget, provider
):
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: Settings(api_key="test-only")))
    monkeypatch.setattr("ragdemo.privacy.build_analyzer", lambda: analyzer)
    monkeypatch.setattr("ragdemo.chunking.TokenBudget.load", classmethod(lambda cls: budget))
    monkeypatch.setattr("ragdemo.provider.OpenRouterProvider", lambda settings: provider)
    original_embed = provider.embed

    def weak_query_embeddings(texts):
        values = original_embed(texts)
        for index, text in enumerate(texts):
            if text.startswith("query: "):
                values[index] = [0.0] * 1024
                values[index][20] = 1
        return values

    monkeypatch.setattr(provider, "embed", weak_query_embeddings)
    app = AppTest.from_file(str(Path("app.py").resolve())).run(timeout=30)
    app.text_area[0].set_value(
        "# Company\n\nVeloNord is a fictional e-bike company with 800 employees."
    )
    app.button[1].click().run(timeout=30)
    provider.response = "VeloNord is a fictional e-bike company with 800 employees. [C0001]"
    app.chat_input[0].set_value("what is velnord?").run(timeout=30)
    assert not app.exception
    assert any("Spelling matches: velnord" in item.value for item in app.text)
    table = app.table[0].value
    assert table.iloc[0]["Lexical matches"] == 1
    assert table.iloc[0]["Matched tokens"] == "velonord"
    app.chat_input[0].set_value("volcanic eruptions").run(timeout=30)
    assert not app.exception
    rejected = app.table[-1].value
    assert rejected.iloc[0]["Lexical matches"] == 0
    assert not rejected.iloc[0]["Used as context"]
    assert "Rejected" in rejected.iloc[0]["Selection reason"]
