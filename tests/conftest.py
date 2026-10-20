import re

import numpy as np
import pytest

from ragdemo.chunking import TokenBudget
from ragdemo.config import Settings
from ragdemo.privacy import PrivacyGuard, build_analyzer
from ragdemo.provider import Completion
from ragdemo.service import RagSession


@pytest.fixture(scope="session")
def analyzer():
    return build_analyzer()


@pytest.fixture(scope="session")
def budget():
    return TokenBudget.load()


class FakeProvider:
    """Deterministic semantic buckets, independent from retrieval ranking logic."""

    def __init__(self):
        self.embedded = []
        self.prompts = []
        self.response = "Refund requests are accepted within 30 days. [C0001]"
        self.error = None

    def embed(self, texts):
        self.embedded.extend(texts)
        if self.error:
            raise self.error
        result = []
        for text in texts:
            vector = np.zeros(1024, dtype=float)
            for axis, pattern in enumerate(
                (r"refund|return|money back", r"ship|deliver|dispatch", r"warranty|repair")
            ):
                if re.search(pattern, text, re.I):
                    vector[axis] = 1
            if not np.any(vector):
                vector[10] = 1
            result.append(vector.tolist())
        return result

    def complete(self, question, chunks):
        self.prompts.append((question, chunks))
        if self.error:
            raise self.error
        return Completion(self.response, 0.01)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def session(analyzer, budget, provider):
    return RagSession(
        Settings(api_key="test-only"), provider, budget, lambda: PrivacyGuard(analyzer)
    )
