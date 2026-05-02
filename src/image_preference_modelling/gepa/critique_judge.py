from __future__ import annotations

import json
from typing import Any

import requests

from image_preference_modelling.config import PromptRewriteModelSettings
from image_preference_modelling.generation_pipeline import DEFAULT_TIMEOUT_SECONDS
from image_preference_modelling.gepa.reward import clamp01


_SYSTEM_PROMPT = (
    "You are a strict evaluator of human image-comparison feedback.\n"
    "You do not decide which image won. The human outcome is authoritative.\n"
    "Your job is to interpret the human critique and estimate how informative and decisive "
    "the feedback is for updating prompt-candidate ratings.\n"
    "The comparison always has a LEFT image and a RIGHT image. Use the provided metadata to "
    "resolve references to left, right, baseline, candidate, first, or second.\n"
    "Return only valid JSON with keys: winner_margin, critique_confidence, winner_evidence, "
    "loser_evidence, tradeoffs, alignment_notes, regression_notes."
)


def fallback_critique_judgement(*, winner: str | None, critique: str) -> dict[str, Any]:
    cleaned = critique.strip()
    detail = min(1.0, len(cleaned) / 120.0)
    if winner is None:
        margin = 0.0
        confidence = 0.35 + 0.35 * detail
    else:
        margin = 0.55 + 0.25 * detail
        confidence = 0.45 + 0.40 * detail
    return {
        "winner_margin": clamp01(margin),
        "critique_confidence": clamp01(confidence),
        "winner_evidence": [],
        "loser_evidence": [],
        "tradeoffs": [cleaned] if cleaned else [],
        "alignment_notes": "Fallback judgement based on critique length.",
        "regression_notes": "",
    }


def judge_critique(
    *,
    original_prompt: str,
    human_winner: str | None,
    left: dict[str, Any],
    right: dict[str, Any],
    critique: str,
    settings: PromptRewriteModelSettings | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    active_settings = settings
    if active_settings is None:
        try:
            active_settings = PromptRewriteModelSettings.from_env()
        except ValueError:
            return fallback_critique_judgement(winner=human_winner, critique=critique)

    payload = {
        "model": active_settings.prompt_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_prompt": original_prompt,
                        "human_outcome": human_winner or "no_clear_winner",
                        "left": left,
                        "right": right,
                        "critique": critique,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        "temperature": 0,
    }
    try:
        response = requests.post(
            f"{active_settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {active_settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = str(response.json()["choices"][0]["message"]["content"]).strip()
        decoded = json.loads(content)
    except (requests.RequestException, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return fallback_critique_judgement(winner=human_winner, critique=critique)

    return _coerce_judgement(decoded, fallback=fallback_critique_judgement(winner=human_winner, critique=critique))


def _coerce_judgement(decoded: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decoded, dict):
        return fallback
    result = dict(fallback)
    result["winner_margin"] = _coerce_float(
        decoded.get("winner_margin"),
        fallback=float(result["winner_margin"]),
    )
    result["critique_confidence"] = _coerce_float(
        decoded.get("critique_confidence"),
        fallback=float(result["critique_confidence"]),
    )
    for key in ("winner_evidence", "loser_evidence", "tradeoffs"):
        value = decoded.get(key)
        if isinstance(value, list):
            result[key] = [str(item).strip() for item in value if str(item).strip()]
    for key in ("alignment_notes", "regression_notes"):
        value = decoded.get(key)
        if isinstance(value, str):
            result[key] = value.strip()
    return result


def _coerce_float(value: Any, *, fallback: float) -> float:
    if isinstance(value, str):
        cleaned = value.strip().removesuffix("%")
        if not cleaned:
            return clamp01(fallback)
        try:
            parsed = float(cleaned)
        except ValueError:
            return clamp01(fallback)
        if parsed > 1.0:
            parsed = parsed / 100.0
        return clamp01(parsed)
    try:
        return clamp01(float(value))
    except (TypeError, ValueError):
        return clamp01(fallback)
