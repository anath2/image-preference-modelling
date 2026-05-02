from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_ELO = 1000.0
DEFAULT_SCORE = 0.5
DEFAULT_CONFIDENCE = 0.0
DEFAULT_TARGET_EVALUATIONS = 5


@dataclass(frozen=True)
class PairwiseEloUpdate:
    left_elo: float
    right_elo: float
    left_result: float
    right_result: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_elo(elo: float, *, floor: float = 800.0, ceiling: float = 1200.0) -> float:
    if ceiling <= floor:
        raise ValueError("ceiling must be greater than floor")
    return clamp01((float(elo) - floor) / (ceiling - floor))


def confidence_from_evidence(
    *,
    evaluation_count: int,
    average_critique_confidence: float,
    target_evaluations: int = DEFAULT_TARGET_EVALUATIONS,
) -> float:
    if target_evaluations < 1:
        raise ValueError("target_evaluations must be >= 1")
    coverage = min(1.0, max(0, int(evaluation_count)) / float(target_evaluations))
    return coverage * clamp01(average_critique_confidence)


def blended_candidate_score(
    *,
    elo: float,
    confidence: float,
    average_margin_quality: float,
) -> float:
    return clamp01(
        0.75 * normalize_elo(elo)
        + 0.15 * clamp01(average_margin_quality)
        + 0.10 * clamp01(confidence)
    )


def pairwise_elo_update(
    *,
    left_elo: float,
    right_elo: float,
    winner: str | None,
    winner_margin: float,
    k_factor: float = 32.0,
) -> PairwiseEloUpdate:
    if winner not in {"left", "right", None}:
        raise ValueError("winner must be 'left', 'right', or None")
    margin = clamp01(winner_margin)
    if winner == "left":
        left_result = 0.5 + 0.5 * margin
    elif winner == "right":
        left_result = 0.5 - 0.5 * margin
    else:
        left_result = 0.5
    right_result = 1.0 - left_result

    expected_left = 1.0 / (1.0 + math.pow(10.0, (float(right_elo) - float(left_elo)) / 400.0))
    expected_right = 1.0 - expected_left
    new_left = float(left_elo) + float(k_factor) * (left_result - expected_left)
    new_right = float(right_elo) + float(k_factor) * (right_result - expected_right)
    return PairwiseEloUpdate(
        left_elo=new_left,
        right_elo=new_right,
        left_result=left_result,
        right_result=right_result,
    )
