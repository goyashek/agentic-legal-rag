"""Tests for the FastAPI layer.

uses FastAPI's TestClient with the agent graph stubbed out, so the API contract tests
run without LLM keys or a live Qdrant.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main, routes
from src.models.schemas import LegalAdvice


def _client() -> TestClient:
    return TestClient(main.app)


class TestQueryEndpoint:
    def test_query_returns_legal_advice_shape(self, monkeypatch) -> None:
        """POST /query returns a LegalAdvice-shaped payload (answer + citations)."""
        monkeypatch.setattr(
            routes,
            "answer_query",
            lambda query: {"answer": LegalAdvice(query=query, answer="BNS 303 applies.")},
        )

        response = _client().post("/query", json={"query": "My bike was stolen"})

        assert response.status_code == 200
        assert response.json()["query"] == "My bike was stolen"
        assert response.json()["answer"] == "BNS 303 applies."

    def test_empty_query_rejected(self) -> None:
        """Empty query -> 422 (QueryRequest min_length=1)."""
        assert _client().post("/query", json={"query": ""}).status_code == 422


class TestHealth:
    def test_health_reports_qdrant_status(self, monkeypatch) -> None:
        monkeypatch.setattr(main, "_qdrant_connected", lambda: True)

        response = _client().get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "qdrant_connected": True, "version": "0.1.0"}

    def test_health_degrades_when_qdrant_is_unavailable(self, monkeypatch) -> None:
        def unavailable() -> bool:
            raise OSError

        monkeypatch.setattr(main, "_qdrant_connected", unavailable)

        response = _client().get("/health")

        assert response.json()["status"] == "degraded"
        assert response.json()["qdrant_connected"] is False
