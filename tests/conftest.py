import os
import time

import jwt
import pytest

TEST_SECRET = "test-secret-that-is-at-least-32-bytes-long"

# Settings are read lazily, so setting env vars here (before any test runs) is enough.
# No real databases or LLMs are touched in the tests.
os.environ.update(
    {
        "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/test",
        "VECTOR_DATABASE_URL": "postgresql+psycopg://test:test@localhost:5433/test",
        "JWT_SECRET": TEST_SECRET,
        "GOOGLE_API_KEY": "not-used",
        "ANTHROPIC_API_KEY": "not-used",
    }
)


@pytest.fixture
def make_token():
    def _make(role: str = "HR", secret: str = TEST_SECRET, expires_in: int = 300) -> str:
        payload = {"sub": "user-1", "role": role, "exp": int(time.time()) + expires_in}
        return jwt.encode(payload, secret, algorithm="HS256")

    return _make
