"""Tests for the FastAPI layer.

uses FastAPI's TestClient with the agent graph stubbed out, so the API contract tests
run without LLM keys or a live Qdrant.
"""

from __future__ import annotations

import pytest


class TestQueryEndpoint:
    @pytest.mark.skip(reason="Week 4 Mon: implement /query first")
    def test_query_returns_legal_advice_shape(self) -> None:
        """POST /query returns a LegalAdvice-shaped payload (answer + citations)."""
        ...

    @pytest.mark.skip(reason="Week 4 Mon")
    def test_empty_query_rejected(self) -> None:
        """Empty query -> 422 (QueryRequest min_length=1)."""
        ...


class TestHealth:
    @pytest.mark.skip(reason="Week 4 Mon: implement /health first")
    def test_health_reports_qdrant_status(self) -> None: ...
