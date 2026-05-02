import pytest

from image_preference_modelling.gepa.critique_judge import _coerce_judgement, fallback_critique_judgement


def test_fallback_critique_judgement_keeps_no_clear_winner_neutral() -> None:
    result = fallback_critique_judgement(
        winner=None,
        critique="Left has stronger composition, but right follows the prompt more closely.",
    )

    assert result["winner_margin"] == 0.0
    assert result["critique_confidence"] > 0.35
    assert result["tradeoffs"]


def test_fallback_critique_judgement_gives_winner_a_margin() -> None:
    result = fallback_critique_judgement(
        winner="right",
        critique="Right follows the prompt more closely while retaining most of the mood.",
    )

    assert result["winner_margin"] > 0.55
    assert result["critique_confidence"] > 0.45


def test_coerce_judgement_falls_back_for_non_numeric_scores() -> None:
    fallback = fallback_critique_judgement(
        winner="right",
        critique="Right is better overall.",
    )

    result = _coerce_judgement(
        {
            "winner_margin": "clear",
            "critique_confidence": "85%",
            "alignment_notes": "right has stronger prompt adherence",
        },
        fallback=fallback,
    )

    assert result["winner_margin"] == fallback["winner_margin"]
    assert result["critique_confidence"] == pytest.approx(0.85)
    assert result["alignment_notes"] == "right has stronger prompt adherence"
