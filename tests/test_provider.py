from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from ragdemo.config import Settings
from ragdemo.errors import DemoError
from ragdemo.models import Chunk
from ragdemo.provider import OpenRouterProvider, _provider_error


class FakeStream:
    def __init__(self, parts, finish="stop", fail=False):
        self.parts, self.finish, self.fail = parts, finish, fail
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def __iter__(self):
        for part in self.parts:
            yield SimpleNamespace(
                model_extra={},
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=part, tool_calls=None, function_call=None),
                        finish_reason=None,
                    )
                ],
            )
        if self.fail:
            raise RuntimeError("raw upstream response alice@example.com")
        if self.finish:
            yield SimpleNamespace(
                model_extra={},
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None, function_call=None),
                        finish_reason=self.finish,
                    )
                ],
            )


def fake_provider(stream):
    provider = OpenRouterProvider(Settings(api_key="test-only"))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return stream

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return provider, calls


def test_stream_buffer_contains_split_pii_only_after_completion():
    stream = FakeStream(["Contact ali", "ce@exa", "mple.com. [C0001]"])
    provider, calls = fake_provider(stream)
    completion = provider.complete("Refund deadline?", [Chunk("C0001", "Reference text", "Policy")])
    assert completion.text == "Contact alice@example.com. [C0001]"
    assert completion.first_token_seconds is not None
    assert stream.closed
    assert calls[0]["stream"] is True
    assert "tools" not in calls[0]
    assert len(calls[0]["messages"]) == 2


@pytest.mark.parametrize("finish,fail", [(None, False), ("length", False), ("stop", True)])
def test_interrupted_or_truncated_stream_never_returns_partial(finish, fail):
    stream = FakeStream(["Contact alice@"], finish=finish, fail=fail)
    provider, _ = fake_provider(stream)
    with pytest.raises(DemoError) as error:
        provider.complete("Refund deadline?", [])
    assert "alice" not in str(error.value)
    assert stream.closed


def test_missing_key_and_safe_provider_error_messages():
    with pytest.raises(DemoError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(Settings(api_key=""))
    for code in (400, 401, 402, 403, 404, 429, 500):
        response = httpx.Response(code, request=httpx.Request("POST", "https://openrouter.ai"))
        error = _provider_error(
            APIStatusError("private alice@example.com", response=response, body=None)
        )
        assert "alice" not in str(error)
    assert "reached" in str(
        _provider_error(APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai")))
    )


def test_exhausted_provider_quota_reports_reset_without_exposing_response_body():
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://openrouter.ai"),
        headers={
            "x-ratelimit-limit": "50",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1792540800000",
        },
    )
    message = str(
        _provider_error(APIStatusError("private upstream content", response=response, body=None))
    )
    assert "0 of 50" in message
    assert "21 October 2026, 02:00 CEST" in message
    assert "private" not in message


def test_malformed_provider_reset_uses_generic_error():
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://openrouter.ai"),
        headers={
            "x-ratelimit-limit": "50",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "not a timestamp",
        },
    )
    message = str(
        _provider_error(APIStatusError("private upstream content", response=response, body=None))
    )
    assert "quota" in message
    assert "private" not in message


def test_streamed_pii_stays_out_of_history_until_full_inspection(session, provider):
    session.ingest(
        text="# Refunds\n\nRefund requests are accepted within 30 days. Email alice@example.com."
    )

    class ObservedStream(FakeStream):
        def __iter__(self):
            for event in super().__iter__():
                assert not session.conversation.messages
                yield event

    stream = ObservedStream(["Refund questions go to ali", "ce@exa", "mple.com. [C0001]"])
    backend, _ = fake_provider(stream)
    backend.embed = provider.embed
    session.provider = backend
    _, answer = session.ask("What is the refund deadline?")
    assert "EMAIL_ADDRESS" in answer.text
    assert "alice@example.com" not in answer.text
    assert len(session.conversation.messages) == 1
    assert stream.closed
