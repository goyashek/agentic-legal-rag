"""Keyless tests for the frontend's small HTTP boundary."""

from __future__ import annotations

from urllib.error import URLError

import pytest

from frontend import app


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_query_api_posts_json(monkeypatch) -> None:
    def fake_urlopen(request, *, timeout):
        assert request.full_url == "http://localhost:8000/query"
        assert request.data == b'{"query": "theft"}'
        assert timeout == 90
        return _Response(b'{"answer": "BNS 303"}')

    monkeypatch.setattr(app, "urlopen", fake_urlopen)

    assert app.query_api("theft") == {"answer": "BNS 303"}


def test_query_api_reports_unavailable_api(monkeypatch) -> None:
    def offline(_request, *, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(app, "urlopen", offline)

    with pytest.raises(RuntimeError, match="Couldn't reach the API"):
        app.query_api("theft")
