"""Tests for the AI research wrapper.

These tests never hit a real LLM — they use ``dry_run=True`` or monkey-patch
vbt internals. They verify the budget + logging + artifact plumbing.
"""

from __future__ import annotations

import json
import os
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.utils import ai_research


@pytest.fixture
def temp_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ai_research to a temp project root for the duration of the test."""

    def fake_root() -> Path:
        return tmp_path

    monkeypatch.setattr(ai_research, "_project_root", fake_root)
    return tmp_path


def _noop_env_load() -> None:
    """Prevent _ensure_env_loaded from reading the real .env during tests."""


def test_pick_provider_prefers_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_research, "_ensure_env_loaded", _noop_env_load)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert ai_research._pick_provider() == "anthropic"


def test_pick_provider_falls_back_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_research, "_ensure_env_loaded", _noop_env_load)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert ai_research._pick_provider() == "openai"


def test_pick_provider_raises_when_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_research, "_ensure_env_loaded", _noop_env_load)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider key"):
        ai_research._pick_provider()


def test_dry_run_does_not_call_provider(
    temp_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    result = ai_research.ask(
        "What is the meaning of life?",
        tag="test",
        dry_run=True,
    )
    assert result.provider == "anthropic"
    assert "DRY RUN" in result.answer
    assert result.estimated_cost_usd == 0.0
    assert result.artifact_path.exists()
    budget_file = temp_project_root / "logs" / "ai_budget.json"
    assert budget_file.exists()
    data = json.loads(budget_file.read_text())
    assert data["call_count"] == 1


def test_budget_cap_is_enforced(
    temp_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    from datetime import datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    budget_file = temp_project_root / "logs" / "ai_budget.json"
    budget_file.parent.mkdir(parents=True, exist_ok=True)
    budget_file.write_text(
        json.dumps({"utc_date": today, "spend_usd": 99.99, "call_count": 5, "queries": []})
    )

    with pytest.raises(ai_research.BudgetExceededError):
        ai_research.ask("Another one", dry_run=True, daily_budget_usd=10.0)


def test_estimate_cost_rough_math() -> None:
    cost = ai_research._estimate_cost("anthropic", in_tok=1_000_000, out_tok=1_000_000)
    # 1M input × $3 + 1M output × $15 = $18
    assert abs(cost - 18.0) < 1e-9


def test_live_path_assertion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a caller whose __file__ sits under src/live/ by patching
    ``sys._getframe`` to return a fake frame chain."""

    live_path = os.path.join("somewhere", "src", "live", "bot.py")

    fake_frame = MagicMock()
    fake_frame.f_code = MagicMock()
    fake_frame.f_code.co_filename = live_path
    fake_frame.f_back = None

    original_getframe = ai_research.sys._getframe

    def patched(depth: int = 0):
        if depth == 1:
            return fake_frame
        return original_getframe(depth)

    monkeypatch.setattr(ai_research.sys, "_getframe", patched)

    with pytest.raises(RuntimeError, match="research-time only"):
        ai_research._assert_not_in_live_path()


def test_budget_rolls_over_day(
    temp_project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget_file = temp_project_root / "logs" / "ai_budget.json"
    budget_file.parent.mkdir(parents=True, exist_ok=True)
    budget_file.write_text(
        json.dumps(
            {"utc_date": "1999-01-01", "spend_usd": 999.0, "call_count": 100, "queries": []}
        )
    )
    state = ai_research.load_budget()
    assert state.spend_usd == 0.0
    assert state.call_count == 0
