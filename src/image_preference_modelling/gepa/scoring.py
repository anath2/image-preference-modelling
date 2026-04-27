from __future__ import annotations

from typing import Any


def _preference_win(outcome: str, winner: str | None) -> float:
    if outcome == "winner":
        if winner == "right":
            return 1.0
        if winner == "left":
            return 0.0
    return 0.5


def _feedback_quality(critique: str) -> float:
    cleaned = critique.strip()
    if not cleaned:
        return 0.0
    return 1.0 if len(cleaned) >= 20 else 0.5


def score_rollout_feedback(rollout_with_feedback: dict[str, Any]) -> dict[str, float]:
    """Score completed rollout feedback using initial M4 objectives."""
    outcome = str(rollout_with_feedback.get("outcome", "")).strip()
    winner = rollout_with_feedback.get("winner")
    critique = str(rollout_with_feedback.get("critique", ""))
    return {
        "preference_win": _preference_win(outcome, winner),
        "feedback_quality": _feedback_quality(critique),
        "intent_preservation": 1.0,
        "composition_preservation": 1.0,
    }
