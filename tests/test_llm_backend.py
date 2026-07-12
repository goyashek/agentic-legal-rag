"""Tests for backend selection + the shared rate limiter in src/agent/llm.py.

The limiter is the sharp one: it's the fix for the grader's 8-way asyncio.gather
burst tripping the free-tier RPM cap (Gemini 15, Cerebras 5). The core guarantee is
that concurrent callers each reserve a distinct, staggered slot instead of all firing
at t=0 — so this proves reservations come out min_interval apart. All keyless, zero quota.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agent import llm


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Two isolations so these tests read ONLY the monkeypatched env, not the machine's:
    (1) clear the module-global limiter/client caches (they key off backend/RPM);
    (2) stub _load_env to a no-op — otherwise its load_dotenv() re-injects the real .env
        (LLM_BACKEND, CEREBRAS_API_KEY, ...) and silently undoes each test's delenv/setenv."""
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    # (3) delenv the override vars a prior test file's real load_dotenv() may have leaked from
    # .env into os.environ (e.g. CEREBRAS_MODEL_FLASH=gemma-4-31b) — they'd shadow the dict
    # defaults these tests assert. Tests that WANT an override just setenv it after this.
    for backend in llm._OPENAI_BACKENDS.values():
        for env_name in backend["model_env"].values():
            monkeypatch.delenv(env_name, raising=False)
    for var in ("LLM_BACKEND", "LLM_RPM", "GEMINI_MODEL_FLASH", "GEMINI_MODEL_PRO"):
        monkeypatch.delenv(var, raising=False)
    llm._LIMITERS.clear()
    llm._SYNC_CLIENTS.clear()
    yield
    llm._LIMITERS.clear()
    llm._SYNC_CLIENTS.clear()


# --- backend selection + model / key resolution (no live calls) ---


def test_openai_backends_resolve_default_model(monkeypatch):
    for backend, flash in [("kiro", "claude-haiku-4.5"), ("groq", "openai/gpt-oss-20b"),
                           ("cerebras", "gpt-oss-120b")]:
        monkeypatch.setenv("LLM_BACKEND", backend)
        assert llm._model_for("flash") == flash


def test_gemini_is_default_backend(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert llm._model_for("flash") == "gemini-3.1-flash-lite"


def test_model_env_override_wins(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "cerebras")
    monkeypatch.setenv("CEREBRAS_MODEL_FLASH", "gpt-oss-20b")
    assert llm._model_for("flash") == "gpt-oss-20b"


def test_has_api_key_per_backend(monkeypatch):
    # kiro has a default gateway key -> always ready
    monkeypatch.setenv("LLM_BACKEND", "kiro")
    assert llm.has_api_key() is True
    # cerebras needs its own key
    monkeypatch.setenv("LLM_BACKEND", "cerebras")
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    assert llm.has_api_key() is False
    monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
    assert llm.has_api_key() is True


def test_openai_client_without_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.get_client("flash")


# --- the rate limiter (the grader-burst fix) ---


def test_reservations_are_staggered_by_interval():
    """Three back-to-back reserves come out ~0, T, 2T apart — this is the burst fix:
    concurrent callers stagger instead of all starting at t=0."""
    lim = llm._RateLimiter(rpm=60)  # min_interval = 1.0s
    waits = [lim._reserve() for _ in range(3)]
    assert waits[0] == pytest.approx(0.0, abs=0.05)
    assert waits[1] == pytest.approx(1.0, abs=0.05)
    assert waits[2] == pytest.approx(2.0, abs=0.05)


def test_rpm_zero_disables_throttle():
    lim = llm._RateLimiter(rpm=0)
    assert lim._reserve() == 0.0
    assert lim._reserve() == 0.0  # kiro is local, no wall


def test_concurrent_async_waits_are_spaced():
    """8 coroutines awaiting wait_async concurrently (the grader's shape) finish spread
    over ~7 intervals, not all at once. Uses a small interval so the test is fast."""
    lim = llm._RateLimiter(rpm=1200)  # 0.05s interval

    async def _run():
        start = asyncio.get_event_loop().time()

        async def _one():
            await lim.wait_async()
            return asyncio.get_event_loop().time() - start

        return await asyncio.gather(*(_one() for _ in range(8)))

    finishes = sorted(asyncio.run(_run()))
    # last one waited ~7 * 0.05 = 0.35s; if they'd all burst it'd be ~0.
    assert finishes[-1] == pytest.approx(0.35, abs=0.1)


def test_limiter_shared_per_backend(monkeypatch):
    """One limiter instance per backend, so the grader burst + pipeline calls draw from
    one global budget (not one bucket per client/tier)."""
    monkeypatch.setenv("LLM_BACKEND", "cerebras")
    monkeypatch.setenv("LLM_RPM", "5")
    a = llm._limiter_for("cerebras")
    b = llm._limiter_for("cerebras")
    assert a is b
    assert a.min_interval == pytest.approx(12.0)  # 60/5
