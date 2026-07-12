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

import os
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "pro"]

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# flash-lite by default: 500 req/day on the free tier vs 20 on 2.5-flash (the
# grader fans out ~8 calls/query, so the daily cap matters). Override per-env with
# GEMINI_MODEL_FLASH / GEMINI_MODEL_PRO. NB: Gemini Pro has no free tier — the pro
# default only works with billing (or point GEMINI_MODEL_PRO at a flash tier).
_DEFAULT_MODELS: dict[Tier, str] = {"flash": "gemini-3.1-flash-lite", "pro": "gemini-2.5-pro"}

# Alternate backend: a local OpenAI-compatible gateway (kiro-gateway) fronting Claude
# via a Kiro subscription. Flip LLM_BACKEND=kiro to route every node through it instead
# of Gemini — no node code changes, since they all call get_client(). Used to escape the
# Gemini free-tier RPM wall for eval runs. Claude models are documented AS Claude (honest
# deviation from D2). TOOLS mode is required: Haiku's JSON mode wraps output in ```json
# fences that instructor can't parse; tool-calling returns clean structured output.
_KIRO_DEFAULT_MODELS: dict[Tier, str] = {"flash": "claude-haiku-4.5", "pro": "claude-sonnet-4.5"}
_KIRO_BASE_URL = "http://localhost:8000/v1"

# Sync clients are safe to reuse process-wide, so cache them. Async clients are
# NOT cached: instructor's google-genai async client must live in the caller's
# event loop, and reusing one across loops raises "attached to a different loop".
_SYNC_CLIENTS: dict[Tier, object] = {}


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

    kiro backend needs no Gemini key (the gateway holds its own auth); gemini needs one.
    """
    if _backend() == "kiro":
        return True
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
    if _backend() == "kiro":
        env_name = "KIRO_MODEL_PRO" if tier == "pro" else "KIRO_MODEL_FLASH"
        return os.getenv(env_name) or _KIRO_DEFAULT_MODELS[tier]
    env_name = "GEMINI_MODEL_PRO" if tier == "pro" else "GEMINI_MODEL_FLASH"
    return os.getenv(env_name) or _DEFAULT_MODELS[tier]


class _ModelBoundClient:
    """Wraps an instructor OpenAI client to inject `model=` into every `.create`.

    The Gemini path bakes the model into `from_provider("google/<model>")`, so nodes call
    `client.create(messages=..., response_model=...)` with NO `model=`. The OpenAI path
    (kiro-gateway) needs `model=` per call. This wrapper supplies it from the tier so the
    six nodes stay backend-agnostic — they never learn which model or backend they're on.
    Forwards the underlying return verbatim, so it works for sync (object) and async
    (coroutine the caller awaits) alike.
    """

    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def create(self, **kwargs):
        kwargs.setdefault("model", self._model)
        return self._client.create(**kwargs)


def _get_kiro_client(tier: Tier, *, async_client: bool):
    """model-bound instructor client over the local kiro-gateway (Claude, OpenAI-compatible).

    TOOLS mode is mandatory (Haiku's JSON mode returns ```json-fenced output instructor
    can't parse). Key is the gateway's own token, not a Gemini key. Sync clients cache;
    async built fresh (same event-loop rule as the Gemini path).
    """
    import instructor
    from openai import AsyncOpenAI, OpenAI

    key = os.getenv("KIRO_API_KEY", "kiro")
    base_url = os.getenv("KIRO_BASE_URL", _KIRO_BASE_URL)
    model = _model_for(tier)
    if async_client:
        raw = instructor.from_openai(
            AsyncOpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        return _ModelBoundClient(raw, model)
    cache_key = f"kiro:{tier}"
    if cache_key not in _SYNC_CLIENTS:
        raw = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        _SYNC_CLIENTS[cache_key] = _ModelBoundClient(raw, model)
    return _SYNC_CLIENTS[cache_key]


def get_client(tier: Tier = "flash", *, async_client: bool = False):
    """Return an instructor-wrapped client for the tier, per the active backend.

    LLM_BACKEND=kiro routes to the local Claude gateway; otherwise Gemini. The model
    string is set inside get_client so nodes stay backend-agnostic. Sync clients are
    cached per tier; async clients are built fresh (they must live in the caller's loop).
    """
    if _backend() == "kiro":
        return _get_kiro_client(tier, async_client=async_client)

    _resolve_key()
    import instructor

    provider = f"google/{_model_for(tier)}"
    if async_client:
        return instructor.from_provider(provider, async_client=True)
    if tier not in _SYNC_CLIENTS:
        _SYNC_CLIENTS[tier] = instructor.from_provider(provider)
    return _SYNC_CLIENTS[tier]


def load_prompt(name: str) -> str:
    """Load a prompt template from agent/prompts/<name>.txt (stripped).

    Not cached on purpose: prompts get tuned during eval and a stale cache would
    make it hard to tell whether a metric moved because of the prompt.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
