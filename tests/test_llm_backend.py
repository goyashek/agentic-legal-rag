"""Keyless tests for DeepSeek client configuration and token ceilings."""

from __future__ import annotations

import asyncio

import pytest

from src.agent import llm


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Keep machine .env values and process-cached clients out of each test."""
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    for var in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL_FLASH",
        "DEEPSEEK_MODEL_PRO",
        "DEEPSEEK_MAX_TOKENS_FLASH",
        "DEEPSEEK_MAX_TOKENS_PRO",
        "DEEPSEEK_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    llm._SYNC_CLIENTS.clear()
    yield
    llm._SYNC_CLIENTS.clear()


def test_default_models_are_deepseek_v4() -> None:
    assert llm._model_for("flash") == "deepseek-v4-flash"
    assert llm._model_for("pro") == "deepseek-v4-pro"


def test_model_and_token_overrides(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL_FLASH", "test-flash")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS_PRO", "777")
    assert llm._model_for("flash") == "test-flash"
    assert llm._max_tokens_for("pro") == 777


def test_non_positive_token_ceiling_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS_FLASH", "0")
    with pytest.raises(ValueError, match="must be positive"):
        llm._max_tokens_for("flash")


def test_timeout_defaults_to_90_seconds_and_can_be_overridden(monkeypatch) -> None:
    assert llm._timeout_seconds() == 90.0
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "12.5")
    assert llm._timeout_seconds() == 12.5


def test_non_positive_timeout_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        llm._timeout_seconds()


def test_key_gating(monkeypatch) -> None:
    assert llm.has_api_key() is False
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert llm.has_api_key() is True


def test_missing_key_raises() -> None:
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm._resolve_key()


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return "ok"


class _AsyncRecorder:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return "ok"


class _AsyncCloseRecorder(_AsyncRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_wrapper_injects_cost_controls() -> None:
    raw = _Recorder()
    client = llm._ClientWrapper(raw, "deepseek-v4-flash", 256, is_async=False)
    assert client.create(messages=[]) == "ok"
    assert raw.kwargs["model"] == "deepseek-v4-flash"
    assert raw.kwargs["max_tokens"] == 256
    assert raw.kwargs["max_retries"] == 0
    assert raw.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_wrapper_preserves_explicit_request_controls() -> None:
    raw = _Recorder()
    client = llm._ClientWrapper(raw, "default", 256, is_async=False)
    client.create(
        model="override",
        max_tokens=42,
        max_retries=2,
        extra_body={"thinking": {"type": "enabled"}},
    )
    assert raw.kwargs["model"] == "override"
    assert raw.kwargs["max_tokens"] == 42
    assert raw.kwargs["max_retries"] == 2
    assert raw.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_async_wrapper_injects_cost_controls() -> None:
    raw = _AsyncRecorder()
    client = llm._ClientWrapper(raw, "deepseek-v4-pro", 1024, is_async=True)
    assert asyncio.run(client.create(messages=[])) == "ok"
    assert raw.kwargs["max_tokens"] == 1024


def test_async_wrapper_closes_its_raw_client() -> None:
    raw = _AsyncCloseRecorder()
    client = llm._ClientWrapper(raw, "deepseek-v4-flash", 256, is_async=True)
    asyncio.run(client.aclose())
    assert raw.closed is True
