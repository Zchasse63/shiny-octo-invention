"""Wrapper around vectorbtpro's Knowledge / Intelligence module.

Provides research-time AI assistance for iterating on strategy design.
Every query is:

1. **Budget-capped** — daily spend ceiling enforced at call time.
2. **Logged** — question, provider, model, token counts, response saved
   to ``events.jsonl`` (kind=``"ai_query"``) AND to a dated Markdown file
   under ``fx-scalper/docs/research/ai_queries/`` so we can grep prior
   conversations.
3. **Provider-switchable** — defaults to Anthropic (user's stated preference;
   better for code-style queries), falls back to OpenAI if Anthropic
   credentials are missing, fails loud if both are missing.
4. **Research-time only** — this module MUST NOT be imported from anything
   in ``src/live/``. It's for iterating on strategies between sessions,
   not for the trading loop. A runtime assertion enforces this.

See ADR 0002 for the three-tier integration plan. This module is Tier 2.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.utils.diary import log_event
from src.utils.logger import get_logger

logger = get_logger(__name__)


Provider = Literal["anthropic", "openai", "gemini"]


# Rough per-1M-token costs (USD). Updated when providers revise.
_COST_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "anthropic": {"input": 3.00, "output": 15.00},    # claude-sonnet-4 ballpark
    "openai": {"input": 2.50, "output": 10.00},       # gpt-4o ballpark
    "gemini": {"input": 0.075, "output": 0.30},       # gemini-1.5-flash ballpark
}


def _project_root() -> Path:
    """Return the fx-scalper project root (one level up from this file's src/)."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class BudgetState:
    """Tracks cumulative spend for the current UTC day.

    Persisted at ``fx-scalper/logs/ai_budget.json`` so the cap survives
    process restarts within a day.
    """

    utc_date: str
    spend_usd: float = 0.0
    call_count: int = 0
    queries: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Outcome of a single AI query.

    Attributes:
        question: The exact question asked.
        answer: Model's response text.
        provider: Which provider answered.
        model: Specific model ID if known.
        input_tokens: Prompt tokens (0 if provider didn't report).
        output_tokens: Completion tokens.
        estimated_cost_usd: Rough cost based on per-million rates.
        artifact_path: Path to the saved Markdown artifact.
    """

    question: str
    answer: str
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    artifact_path: Path


# ---------------------------------------------------------------------------
# Module guard — refuse import from src/live/
# ---------------------------------------------------------------------------


def _assert_not_in_live_path() -> None:
    """Runtime check that this module isn't being imported by live-trading code.

    Walks the call stack looking for files under ``src/live/``. Raises
    ``RuntimeError`` if found. Cheap enough to run per-call; makes the
    "Tier 2, research only" contract enforceable.
    """
    frame = sys._getframe(1)
    while frame is not None:
        caller_path = frame.f_code.co_filename
        if os.sep + "src" + os.sep + "live" + os.sep in caller_path:
            raise RuntimeError(
                f"src.utils.ai_research used by live-trading code at "
                f"{caller_path}. This is Tier 2 research-time only. "
                f"See ADR 0002."
            )
        frame = frame.f_back


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def _budget_path() -> Path:
    p = _project_root() / "logs" / "ai_budget.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_budget() -> BudgetState:
    """Load today's budget state from disk; fresh state if date rolled over."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    path = _budget_path()
    if not path.exists():
        return BudgetState(utc_date=today)
    try:
        data = json.loads(path.read_text())
    except Exception:
        return BudgetState(utc_date=today)
    if data.get("utc_date") != today:
        return BudgetState(utc_date=today)
    return BudgetState(
        utc_date=data["utc_date"],
        spend_usd=float(data.get("spend_usd", 0.0)),
        call_count=int(data.get("call_count", 0)),
        queries=list(data.get("queries", [])),
    )


def save_budget(state: BudgetState) -> None:
    """Persist budget state to disk."""
    path = _budget_path()
    path.write_text(
        json.dumps(
            {
                "utc_date": state.utc_date,
                "spend_usd": state.spend_usd,
                "call_count": state.call_count,
                "queries": state.queries[-100:],  # keep last 100 for context
            },
            indent=2,
            default=str,
        )
    )


DAILY_BUDGET_USD_DEFAULT: float = 10.0
"""Daily spend cap. Exceeding raises :class:`BudgetExceededError`."""


class BudgetExceededError(RuntimeError):
    """Raised when a query would push daily spend above the cap."""


# ---------------------------------------------------------------------------
# Artifact storage
# ---------------------------------------------------------------------------


def _artifact_dir() -> Path:
    p = _project_root() / "docs" / "research" / "ai_queries"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_artifact(
    question: str,
    answer: str,
    provider: Provider,
    model: str,
    tag: str | None,
    tokens: dict[str, int],
    cost_usd: float,
) -> Path:
    """Persist a Q+A pair to docs/research/ai_queries/<ts>-<tag>.md."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    slug = (tag or "query").replace(" ", "_").replace("/", "_")[:40]
    path = _artifact_dir() / f"{ts}-{slug}.md"
    content = f"""# {tag or 'AI query'}

**When:** {datetime.now(UTC).isoformat(timespec='seconds')} UTC
**Provider:** {provider} / `{model}`
**Tokens:** input={tokens.get('input', 0)} output={tokens.get('output', 0)}
**Estimated cost:** ${cost_usd:.4f}

## Question

{question}

## Answer

{answer}
"""
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _ensure_env_loaded() -> None:
    """Eagerly load ``fx-scalper/.env`` into the process env.

    ``python -c`` entry points don't always have the right CWD; we resolve
    the .env path relative to this module's file location so it always works.
    Uses ``override=True`` because the shell may pre-set API key variables
    to empty strings (python-dotenv's default ``override=False`` treats an
    empty-string env var as "already set" and skips).
    """
    from dotenv import load_dotenv

    env_path = _project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def _pick_provider(override: Provider | None = None) -> Provider:
    """Return the first available provider from env keys. Raises if none."""
    if override is not None:
        return override
    _ensure_env_loaded()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    raise RuntimeError(
        "No LLM provider key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "in fx-scalper/.env."
    )


def _estimate_cost(provider: Provider, in_tok: int, out_tok: int) -> float:
    rates = _COST_PER_MILLION_TOKENS.get(provider, {"input": 0.0, "output": 0.0})
    return (in_tok / 1_000_000) * rates["input"] + (out_tok / 1_000_000) * rates["output"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ask(
    question: str,
    *,
    tag: str | None = None,
    provider: Provider | None = None,
    daily_budget_usd: float = DAILY_BUDGET_USD_DEFAULT,
    quick: bool = False,
    dry_run: bool = False,
) -> QueryResult:
    """Ask vectorbtpro's Knowledge module a question.

    Args:
        question: The question in plain English. Use full sentences + context;
            vbt.chat uses RAG over the vbtpro docs/examples corpus so specifics
            help retrieval.
        tag: Optional short label used in the artifact filename + event log.
            E.g. ``"pullback_ema_next_iteration"``.
        provider: Force a specific provider. Default: auto-pick from env.
        daily_budget_usd: Cap on cumulative spend for the UTC day. Default $10.
        quick: Pass ``quick_mode=True`` to vbt — shorter, faster, cheaper.
        dry_run: If True, don't actually call the LLM. Useful for testing the
            logging + budget path without burning tokens.

    Returns:
        :class:`QueryResult`.

    Raises:
        BudgetExceededError: If this call would push daily spend over cap.
        RuntimeError: If no provider is configured or vbtpro import fails.
    """
    _assert_not_in_live_path()

    chosen = _pick_provider(provider)
    budget = load_budget()
    if budget.spend_usd >= daily_budget_usd:
        raise BudgetExceededError(
            f"Daily AI budget ${daily_budget_usd:.2f} already consumed "
            f"(${budget.spend_usd:.2f}). Reset at 00:00 UTC."
        )

    if dry_run:
        answer = f"[DRY RUN] Would have asked ({chosen}): {question[:200]}"
        tokens = {"input": len(question.split()), "output": 0}
        cost = 0.0
        model = f"{chosen}/dry-run"
    else:
        answer, tokens, model = _call_vbt_chat(question, chosen, quick=quick)
        cost = _estimate_cost(chosen, tokens.get("input", 0), tokens.get("output", 0))

    # Advisory check — stop if this call crossed the cap after the fact.
    budget.spend_usd += cost
    budget.call_count += 1
    budget.queries.append(
        {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "tag": tag,
            "provider": chosen,
            "tokens": tokens,
            "cost_usd": cost,
        }
    )
    save_budget(budget)

    artifact_path = _save_artifact(
        question=question,
        answer=answer,
        provider=chosen,
        model=model,
        tag=tag,
        tokens=tokens,
        cost_usd=cost,
    )

    log_event(
        "ai_query",
        tag=tag,
        provider=chosen,
        model=model,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        cost_usd=round(cost, 6),
        artifact=str(artifact_path),
        daily_spend_so_far_usd=round(budget.spend_usd, 4),
    )

    if budget.spend_usd >= daily_budget_usd:
        logger.warning(
            f"AI budget cap reached: ${budget.spend_usd:.2f} of "
            f"${daily_budget_usd:.2f}. Further calls until 00:00 UTC will raise."
        )

    return QueryResult(
        question=question,
        answer=answer,
        provider=chosen,
        model=model,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        estimated_cost_usd=cost,
        artifact_path=artifact_path,
    )


def search(
    query: str,
    *,
    tag: str | None = None,
    top_k: int = 10,
) -> str:
    """Retrieval-only search over the vectorbtpro docs/examples corpus.

    No LLM completion — just returns the top-K ranked excerpts. Cheap
    (embeddings cached after first call) and deterministic.

    Args:
        query: What you're searching for.
        tag: Optional label for the event log.
        top_k: How many excerpts to return.

    Returns:
        HTML-formatted search results string.
    """
    _assert_not_in_live_path()
    try:
        import vectorbtpro as vbt
    except ImportError as e:
        raise RuntimeError(f"vectorbtpro not installed: {e}") from e

    logger.info(f"vbt.search({query!r}, top_k={top_k})")
    result = vbt.search(query, top_k=top_k)

    log_event(
        "ai_search",
        tag=tag,
        query=query,
        top_k=top_k,
        result_length=len(str(result)) if result is not None else 0,
    )
    return str(result) if result is not None else ""


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _call_vbt_chat(
    question: str,
    provider: Provider,
    *,
    quick: bool,
) -> tuple[str, dict[str, int], str]:
    """Invoke ``vbt.chat`` with the given provider, return (answer, tokens, model).

    Returns tokens as best-effort; vbt doesn't always surface them. If not
    available, returns rough word-count estimates.
    """
    try:
        import vectorbtpro as vbt
    except ImportError as e:
        raise RuntimeError(f"vectorbtpro not installed: {e}") from e

    # vbt respects env vars for provider API keys; we make sure the right
    # one is active (the others are fine to leave set).
    # No explicit vbt config here — it auto-selects based on what's in env.

    logger.info(f"vbt.chat(provider={provider}, quick={quick}): {question[:80]}…")

    # stream=False so the result is returned as a string, not printed to stdout
    # incrementally. return_chat=True gives us (response, Chat) — Chat exposes
    # chat_history from which we can extract the final assistant turn reliably.
    kwargs: dict[str, Any] = {
        "quick_mode": quick,
        "stream": False,
        "return_chat": True,
    }

    raw = vbt.chat(question, **kwargs)

    # vbt.chat with return_chat=True returns (response, chat). Extract the
    # assistant message from chat_history as the canonical answer.
    answer = ""
    chat = None
    if isinstance(raw, tuple) and len(raw) == 2:
        response, chat = raw
        if isinstance(response, str):
            answer = response
    elif isinstance(raw, str):
        answer = raw

    if not answer and chat is not None:
        history = getattr(chat, "chat_history", None) or []
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                answer = content if isinstance(content, str) else str(content)
                break

    in_tokens = len(question) // 4  # ~4 chars per token
    out_tokens = len(answer) // 4
    model = f"{provider}/default"
    return answer, {"input": in_tokens, "output": out_tokens}, model
