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

# Sync clients are safe to reuse process-wide, so cache them. Async clients are
# NOT cached: instructor's google-genai async client must live in the caller's
# event loop, and reusing one across loops raises "attached to a different loop".
_SYNC_CLIENTS: dict[Tier, object] = {}


def _load_env() -> None:
    """Load .env into the environment (idempotent, no-op if the file is absent)."""
    from dotenv import load_dotenv

    load_dotenv()


def has_api_key() -> bool:
    """True if a Gemini key is available. Gate live-LLM tests on this."""
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
    env_name = "GEMINI_MODEL_PRO" if tier == "pro" else "GEMINI_MODEL_FLASH"
    return os.getenv(env_name) or _DEFAULT_MODELS[tier]


def get_client(tier: Tier = "flash", *, async_client: bool = False):
    """Return an instructor-wrapped Gemini client for the tier.

    Sync clients are cached per tier; async clients are built fresh (they must
    live in the caller's event loop). Raises if no API key is configured.
    """
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
