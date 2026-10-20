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
