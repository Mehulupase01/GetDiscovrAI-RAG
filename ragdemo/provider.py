from dataclasses import dataclass
from time import monotonic

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from ragdemo.config import Settings
from ragdemo.errors import DemoError
from ragdemo.models import Chunk
from ragdemo.safety import BASIC_SYSTEM_PROMPT, SYSTEM_PROMPT


@dataclass(frozen=True)
class Completion:
    text: str
    first_token_seconds: float | None


def _provider_error(exc: Exception) -> DemoError:
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return DemoError(
                "OpenRouter rejected access. Check your API key and model permissions."
            )
        if exc.status_code == 429:
            return DemoError(
                "OpenRouter's free-provider quota is busy or exhausted. Try again later."
            )
        if exc.status_code == 402:
            return DemoError(
                "OpenRouter requires credits or an account setting change for this request."
            )
        if exc.status_code in (400, 404):
            return DemoError(
                "OpenRouter could not serve the configured model or request. Check the model ID and availability."
            )
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return DemoError("OpenRouter could not be reached in time. Check connectivity and retry.")
    return DemoError("The model request failed or its response was incomplete. Please retry.")


class OpenRouterProvider:
    def __init__(self, settings: Settings):
        settings.require_key()
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=45.0,
            max_retries=1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = []
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                response = self._client.embeddings.create(
                    model=self.settings.embedding_model, input=batch, encoding_format="float"
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in ordered] != list(range(len(batch))):
                    raise DemoError("The embedding provider returned an incomplete batch.")
                vectors.extend(item.embedding for item in ordered)
            return vectors
        except DemoError:
            raise
        except Exception as exc:
            raise _provider_error(exc) from None

    def complete(self, question: str, chunks: list[Chunk]) -> Completion:
        # JSON quoting makes the data envelope unambiguous even with hostile delimiters.
        import json

        evidence = json.dumps(
            [{"id": chunk.id, "section": chunk.section, "text": chunk.text} for chunk in chunks],
            ensure_ascii=False,
        )
        started = monotonic()
        first_token = None
        parts: list[str] = []
        size = 0
        finished = False
        try:
            with self._client.chat.completions.create(
                model=self.settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        if self.settings.safety_checks_enabled
                        else BASIC_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "evidence": json.loads(evidence)},
                            ensure_ascii=False,
                        ),
                    },
                ],
                stream=True,
                max_tokens=1_000,
                temperature=0,
            ) as stream:
                for event in stream:
                    if monotonic() - started > 90 or (event.model_extra or {}).get("error"):
                        raise DemoError("The model response was interrupted. Please retry.")
                    for choice in event.choices:
                        if choice.delta.tool_calls or choice.delta.function_call:
                            raise DemoError(
                                "The answer was withheld because the model attempted a tool call."
                            )
                        content = choice.delta.content
                        if content:
                            if first_token is None:
                                first_token = monotonic() - started
                            parts.append(content)
                            size += len(content)
                            if size > 16_000:
                                raise DemoError("The model response exceeded the safe answer size.")
                        if choice.finish_reason is not None:
                            if choice.finish_reason != "stop":
                                raise DemoError(
                                    "The model response was incomplete and was discarded. Please retry."
                                )
                            finished = True
            text = "".join(parts).strip()
            if not finished or not text:
                raise DemoError(
                    "The model response was incomplete and was discarded. Please retry."
                )
            return Completion(text, first_token)
        except DemoError:
            raise
        except Exception as exc:
            raise _provider_error(exc) from None
