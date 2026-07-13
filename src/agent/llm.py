"""Shared DeepSeek access for the agent's structured-output nodes.

Flash handles routing, grading, checking, rewriting, and intent expansion; Pro
generates the final answer. Every DeepSeek call runs in non-thinking mode: these
are bounded classification/extraction tasks, and the default thinking mode spends
tokens without improving the structured result.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Literal

Tier = Literal["flash", "pro"]

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MODELS: dict[Tier, str] = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}
# Safe ceilings for the Pydantic schemas used here: compact control-plane calls
# get 256 tokens; the user-facing cited answer gets 1024.
_MAX_TOKENS: dict[Tier, int] = {"flash": 256, "pro": 1024}
_SYNC_CLIENTS: dict[Tier, object] = {}


def _load_env() -> None:
    """Load .env into the environment (idempotent, no-op if absent)."""
    from dotenv import load_dotenv

    load_dotenv()


def has_api_key() -> bool:
    """True when the DeepSeek key is available for an opt-in live test."""
    _load_env()
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def _resolve_key() -> str:
    """Return the DeepSeek key or fail at the LLM boundary."""
    _load_env()
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM nodes need DEEPSEEK_API_KEY. The deterministic nodes "
            "(fast_path, ood_gate, citation_validator) run without a key."
        )
    return key


def _model_for(tier: Tier) -> str:
    _load_env()
    return os.getenv(f"DEEPSEEK_MODEL_{tier.upper()}") or _MODELS[tier]


def _max_tokens_for(tier: Tier) -> int:
    """Return a positive, per-tier completion ceiling from the environment."""
    _load_env()
    value = int(os.getenv(f"DEEPSEEK_MAX_TOKENS_{tier.upper()}", _MAX_TOKENS[tier]))
    if value < 1:
        raise ValueError("DEEPSEEK_MAX_TOKENS values must be positive")
    return value


class _ClientWrapper:
    """Inject DeepSeek's model, non-thinking mode, and output ceiling once."""

    def __init__(self, client, model: str, max_tokens: int, is_async: bool) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._is_async = is_async

    def create(self, **kwargs):
        kwargs.setdefault("model", self._model)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        if self._is_async:
            return self._acreate(**kwargs)
        return self._client.create(**kwargs)

    async def _acreate(self, **kwargs):
        return await self._client.create(**kwargs)

    async def aclose(self) -> None:
        """Close an owned async instructor client before its event loop ends."""
        if not self._is_async:
            return
        result = self._client.close()
        if inspect.isawaitable(result):
            await result


def get_client(tier: Tier = "flash", *, async_client: bool = False):
    """Return an instructor-wrapped DeepSeek client for ``tier``.

    Async clients are deliberately fresh because they belong to the caller's event
    loop. Sync clients are process-cached.
    """
    import instructor
    from openai import AsyncOpenAI, OpenAI

    key = _resolve_key()
    model = _model_for(tier)
    max_tokens = _max_tokens_for(tier)
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if async_client:
        raw = instructor.from_openai(
            AsyncOpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        return _ClientWrapper(raw, model, max_tokens, is_async=True)
    if tier not in _SYNC_CLIENTS:
        raw = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=key), mode=instructor.Mode.TOOLS
        )
        _SYNC_CLIENTS[tier] = _ClientWrapper(raw, model, max_tokens, is_async=False)
    return _SYNC_CLIENTS[tier]


def load_prompt(name: str) -> str:
    """Load a prompt template from agent/prompts/<name>.txt (stripped)."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
