"""Shared Gemini access for the agent's LLM nodes.

One place to get an `instructor`-wrapped Gemini client, so the six LLM nodes
(router, intent_expander, grader, rewriter, generator, checker) don't each
re-implement key gating and provider wiring. Two tiers, per D2:

    flash -> routing / grading / checking / rewriting / intent expansion
    pro   -> answer generation

Key gating matches `chunk_chonkie.summarize_section`: needs GOOGLE_API_KEY or
GEMINI_API_KEY and raises loudly if absent, so a keyless run fails here instead
of degrading silently. The `instructor` google-genai backend reads
GOOGLE_API_KEY, so a GEMINI_API_KEY-only env gets mirrored across.

Injectable by design: every node takes an optional `client` and falls back to
`get_client(tier)`. Unit tests pass a fake client whose `.create(...)` returns a
canned Pydantic object, so node logic is exercised at zero quota; live-Gemini
tests gate on `has_api_key()`.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "pro"]

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# flash-lite by default: 500 req/day on the free tier vs 20 on 2.5-flash (the
# grader fans out ~8 calls/query, so the daily cap matters). Override per-env with
# GEMINI_MODEL_FLASH / GEMINI_MODEL_PRO. NB: Gemini Pro has no free tier — the pro
# default only works with billing (or point GEMINI_MODEL_PRO at a flash tier).
_DEFAULT_MODELS: dict[Tier, str] = {"flash": "gemini-3.1-flash-lite", "pro": "gemini-2.5-pro"}

# OpenAI-compatible backends: same instructor.from_openai path, differing only by
# base_url / key env / models. LLM_BACKEND selects one; nodes are unchanged (they all
# call get_client). All route Claude-or-GPT-OSS AS that model (honest deviation from D2 —
# never relabel these as Gemini). TOOLS mode is required across the board: Haiku's JSON
# mode wraps output in ```json fences instructor can't parse; tool-calling is clean.
#   kiro     — local gateway fronting Claude via a Kiro sub (no RPM wall; per-call token cost)
#   groq     — GPT-OSS 20b/120b, strict structured output (free tier: 30 RPM but 8K TPM is the
#              real wall on the grader's big prompts; RPM limiting is coarse, bill for full runs)
#   cerebras — GPT-OSS 120b, 5 RPM free (RPM limiter fixes the grader burst here cleanly)
_OPENAI_BACKENDS: dict[str, dict] = {
    "kiro": {
        "base_url": "http://localhost:8000/v1",
        "base_url_env": "KIRO_BASE_URL",
        "key_env": "KIRO_API_KEY",
        "default_key": "kiro",
        "model_env": {"flash": "KIRO_MODEL_FLASH", "pro": "KIRO_MODEL_PRO"},
        "models": {"flash": "claude-haiku-4.5", "pro": "claude-sonnet-4.5"},
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "base_url_env": "GROQ_BASE_URL",
        "key_env": "GROQ_API_KEY",
        "default_key": None,  # required
        "model_env": {"flash": "GROQ_MODEL_FLASH", "pro": "GROQ_MODEL_PRO"},
        "models": {"flash": "openai/gpt-oss-20b", "pro": "openai/gpt-oss-120b"},
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "base_url_env": "CEREBRAS_BASE_URL",
        "key_env": "CEREBRAS_API_KEY",
        "default_key": None,  # required
        "model_env": {"flash": "CEREBRAS_MODEL_FLASH", "pro": "CEREBRAS_MODEL_PRO"},
        "models": {"flash": "gpt-oss-120b", "pro": "gpt-oss-120b"},
    },
    "deepseek": {
        # Paid credits, OpenAI-compatible, tool-calling confirmed (supportsToolCall: true) so
        # instructor TOOLS mode works. NO free-tier RPM wall (throttles dynamically under load,
        # no hard cap) — this is the backend that escapes the burst/capacity problem. NB: the
        # deepseek-chat/deepseek-reasoner aliases deprecate 2026-07-24; use the v4 IDs.
        "base_url": "https://api.deepseek.com",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "key_env": "DEEPSEEK_API_KEY",
        "default_key": None,  # required
        "model_env": {"flash": "DEEPSEEK_MODEL_FLASH", "pro": "DEEPSEEK_MODEL_PRO"},
        "models": {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"},
        # v4 defaults to thinking mode, which rejects the tool_choice instructor's TOOLS mode
        # sets ("Thinking mode does not support this tool_choice"). Our structured nodes are
        # classification/extraction — no reasoning needed — so disable thinking on every call.
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}

# Free-tier RPM per backend — the grader fires ~8 concurrent .create calls, which bursts
# straight past these caps unless spaced. The limiter (below) reads LLM_RPM (override) or
# these defaults. 0 = no throttle (kiro is local, no RPM wall). Groq's real limit is TPM,
# not RPM; an RPM value here is a coarse fit — set LLM_RPM low or bill for full Groq runs.
_DEFAULT_RPM: dict[str, int] = {"gemini": 15, "cerebras": 5, "groq": 30, "kiro": 0, "deepseek": 0}

# Sync clients are safe to reuse process-wide, so cache them. Async clients are
# NOT cached: instructor's google-genai async client must live in the caller's
# event loop, and reusing one across loops raises "attached to a different loop".
_SYNC_CLIENTS: dict[str, object] = {}

# One limiter per backend, shared across every client/tier so the RPM budget is global
# (the grader's burst + the pipeline's ~12 calls/query all draw from the same bucket).
_LIMITERS: dict[str, _RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


class _RateLimiter:
    """Global min-interval gate: spaces call starts >= 60/rpm apart.

    State is monotonic-time in a threading.Lock (NOT an asyncio primitive), so it survives
    across separate asyncio.run() loops (RAGAS runs one loop per scenario) and works for
    sync + concurrent-async alike. Each caller reserves a distinct future slot in the tiny
    lock-held critical section, then sleeps outside the lock — so 8 concurrent grader
    coroutines fan their reservations 0, T, 2T... and fire spaced instead of bursting.
    rpm <= 0 disables throttling (min_interval 0 -> no wait).
    """

    def __init__(self, rpm: float) -> None:
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0  # earliest monotonic time the next call may start

    def _reserve(self) -> float:
        """Claim the next slot; return seconds to wait before the call may start."""
        if self.min_interval <= 0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
            return start - now

    def wait(self) -> None:
        w = self._reserve()
        if w > 0:
            time.sleep(w)

    async def wait_async(self) -> None:
        w = self._reserve()
        if w > 0:
            await asyncio.sleep(w)


def _limiter_for(backend: str) -> _RateLimiter:
    """Shared limiter for the backend; LLM_RPM overrides the per-backend default RPM."""
    with _LIMITERS_LOCK:
        if backend not in _LIMITERS:
            rpm_env = os.getenv("LLM_RPM")
            rpm = int(rpm_env) if rpm_env else _DEFAULT_RPM.get(backend, 0)
            _LIMITERS[backend] = _RateLimiter(rpm)
        return _LIMITERS[backend]


def _load_env() -> None:
    """Load .env into the environment (idempotent, no-op if the file is absent)."""
    from dotenv import load_dotenv

    load_dotenv()


def _backend() -> str:
    """Which LLM backend to use: 'kiro' (local Claude gateway) or 'gemini' (default)."""
    _load_env()
    return os.getenv("LLM_BACKEND", "gemini").strip().lower()


def has_api_key() -> bool:
    """True if the configured backend can make a live call. Gate live-LLM tests on this.

    kiro holds its own gateway auth (default key), so it's always ready; groq/cerebras
    need their own key set; gemini needs a Google/Gemini key.
    """
    backend = _backend()
    if backend in _OPENAI_BACKENDS:
        cfg = _OPENAI_BACKENDS[backend]
        _load_env()
        return bool(os.getenv(cfg["key_env"]) or cfg["default_key"])
    _load_env()
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _resolve_key() -> str:
    """Return the Gemini key, mirroring it to GOOGLE_API_KEY for google-genai.

    Raises RuntimeError if neither name is set, matching summarize_section so a
    keyless run fails at the LLM boundary rather than silently degrading.
    """
    _load_env()
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM nodes need GOOGLE_API_KEY or GEMINI_API_KEY. Set one in .env "
            "(see .env.example). The deterministic nodes (fast_path, ood_gate, "
            "citation_validator) run without a key."
        )
    # google-genai reads GOOGLE_API_KEY specifically; mirror a GEMINI-only env.
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return key


def _model_for(tier: Tier) -> str:
    backend = _backend()
    if backend in _OPENAI_BACKENDS:
        cfg = _OPENAI_BACKENDS[backend]
        return os.getenv(cfg["model_env"][tier]) or cfg["models"][tier]
    env_name = "GEMINI_MODEL_PRO" if tier == "pro" else "GEMINI_MODEL_FLASH"
    return os.getenv(env_name) or _DEFAULT_MODELS[tier]


class _ClientWrapper:
    """Wraps an instructor client to (1) space calls through the backend's rate limiter
    and (2) inject `model=` when the client needs it (the OpenAI path).

    The Gemini path bakes the model into `from_provider("google/<model>")`, so `model` is
    None and nothing is injected. The OpenAI backends (kiro/groq/cerebras) need `model=`
    per call, supplied from the tier — so the six nodes stay backend-agnostic, never
    learning which model or backend they're on.

    Limiting happens HERE, at the one chokepoint every node's call flows through, so the
    grader's 8-way asyncio.gather burst gets spaced to the RPM floor without any node
    change. Sync `.create` waits synchronously; async returns a coroutine that awaits the
    async wait first, then the underlying create — so concurrent coroutines each reserve a
    distinct slot up front and fire staggered instead of all at once.
    """

    def __init__(
        self,
        client,
        model: str | None,
        limiter: _RateLimiter,
        is_async: bool,
        extra_body: dict | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._limiter = limiter
        self._is_async = is_async
        self._extra_body = extra_body  # per-backend default (e.g. deepseek thinking:disabled)

    def create(self, **kwargs):
        if self._model is not None:
            kwargs.setdefault("model", self._model)
        if self._extra_body is not None:
            kwargs.setdefault("extra_body", self._extra_body)
        if self._is_async:
            return self._acreate(**kwargs)
        self._limiter.wait()
        return self._client.create(**kwargs)

    async def _acreate(self, **kwargs):
        await self._limiter.wait_async()
        return await self._client.create(**kwargs)


def _get_openai_client(backend: str, tier: Tier, *, async_client: bool):
    """Limiter-wrapped instructor client over an OpenAI-compatible backend (kiro/groq/cerebras).

    TOOLS mode is mandatory across all three (Haiku's JSON mode returns ```json-fenced
    output instructor can't parse; GPT-OSS is happy in TOOLS too). base_url/key/models come
    from _OPENAI_BACKENDS, overridable per-env. Sync clients cache; async built fresh
    (instructor's async client must live in the caller's event loop).
    """
    import instructor
    from openai import AsyncOpenAI, OpenAI

    cfg = _OPENAI_BACKENDS[backend]
    _load_env()
    key = os.getenv(cfg["key_env"]) or cfg["default_key"]
    if not key:
        raise RuntimeError(
            f"LLM_BACKEND={backend} needs {cfg['key_env']} set in .env (see .env.example)."
        )
    base_url = os.getenv(cfg["base_url_env"]) or cfg["base_url"]
    model = _model_for(tier)
    limiter = _limiter_for(backend)
    extra_body = cfg.get("extra_body")
    if async_client:
        raw = instructor.from_openai(
            AsyncOpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        return _ClientWrapper(raw, model, limiter, is_async=True, extra_body=extra_body)
    cache_key = f"{backend}:{tier}"
    if cache_key not in _SYNC_CLIENTS:
        raw = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        _SYNC_CLIENTS[cache_key] = _ClientWrapper(
            raw, model, limiter, is_async=False, extra_body=extra_body
        )
    return _SYNC_CLIENTS[cache_key]


def get_client(tier: Tier = "flash", *, async_client: bool = False):
    """Return a rate-limited, instructor-wrapped client for the tier, per the active backend.

    LLM_BACKEND selects: kiro/groq/cerebras (OpenAI-compatible) or gemini (default). Every
    call is spaced through a shared per-backend rate limiter (fixes the grader burst). Model
    is set inside get_client so nodes stay backend-agnostic. Sync clients cache per
    backend+tier; async clients are built fresh (they must live in the caller's loop).
    """
    backend = _backend()
    if backend in _OPENAI_BACKENDS:
        return _get_openai_client(backend, tier, async_client=async_client)

    _resolve_key()
    import instructor

    limiter = _limiter_for("gemini")
    provider = f"google/{_model_for(tier)}"
    if async_client:
        raw = instructor.from_provider(provider, async_client=True)
        return _ClientWrapper(raw, None, limiter, is_async=True)
    cache_key = f"gemini:{tier}"
    if cache_key not in _SYNC_CLIENTS:
        raw = instructor.from_provider(provider)
        _SYNC_CLIENTS[cache_key] = _ClientWrapper(raw, None, limiter, is_async=False)
    return _SYNC_CLIENTS[cache_key]


def load_prompt(name: str) -> str:
    """Load a prompt template from agent/prompts/<name>.txt (stripped).

    Not cached on purpose: prompts get tuned during eval and a stale cache would
    make it hard to tell whether a metric moved because of the prompt.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
