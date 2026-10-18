from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_graph


@pytest.fixture
def fake_graph():
    graph = MagicMock()
    graph.invoke.return_value = {
        "answer": "You get 25 days of leave.",
        "route": "docs",
        "sources": [{"filename": "leave.pdf", "chunk": 0}],
        "models": {"classify": "claude-haiku-5-5", "answer": "gemini-2.5-flash"},
    }
    app.dependency_overrides[get_graph] = lambda: graph
    yield graph
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_without_token_returns_401(fake_graph):
    response = client.post("/ask", json={"question": "How many leave days do I get?"})
    assert response.status_code == 401
    fake_graph.invoke.assert_not_called()


def test_ask_with_bad_signature_returns_401(fake_graph, make_token):
    token = make_token(secret="a-different-secret-also-32-bytes-long")
    response = client.post(
        "/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


def test_ask_with_expired_token_returns_401(fake_graph, make_token):
    token = make_token(expires_in=-10)
    response = client.post(
        "/ask", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


def test_ask_with_valid_token_passes_role_to_graph(fake_graph, make_token):
    token = make_token(role="Employee")
    response = client.post(
        "/ask",
        json={"question": "How many leave days do I get?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "You get 25 days of leave.",
        "route": "docs",
        "sources": [{"filename": "leave.pdf", "chunk": 0}],
        "models": {"classify": "claude-haiku-5-5", "answer": "gemini-2.5-flash"},
    }
    fake_graph.invoke.assert_called_once_with(
        {"question": "How many leave days do I get?", "user_role": "Employee"}
    )


def test_ingest_requires_hr(make_token):
    token = make_token(role="Employee")
    response = client.post(
        "/ingest",
        files={"file": ("policy.txt", b"Some policy", "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
