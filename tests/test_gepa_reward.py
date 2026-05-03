from image_preference_modelling.gepa.reward import (
    blended_candidate_score,
    confidence_from_evidence,
    pairwise_elo_update,
)


def test_pairwise_elo_update_uses_margin_without_flipping_human_winner() -> None:
    update = pairwise_elo_update(
        left_elo=1000.0,
        right_elo=1000.0,
        winner="right",
        winner_margin=0.5,
    )

    assert update.right_elo > 1000.0
    assert update.left_elo < 1000.0
    assert update.right_result == 0.75
    assert update.left_result == 0.25


def test_pairwise_elo_update_no_clear_winner_is_neutral() -> None:
    update = pairwise_elo_update(
        left_elo=1000.0,
        right_elo=1000.0,
        winner=None,
        winner_margin=1.0,
    )

    assert update.left_elo == 1000.0
    assert update.right_elo == 1000.0
    assert update.left_result == 0.5
    assert update.right_result == 0.5


def test_confidence_and_blended_score_are_bounded() -> None:
    confidence = confidence_from_evidence(evaluation_count=3, average_critique_confidence=0.8)
    score = blended_candidate_score(elo=1100.0, confidence=confidence, average_margin_quality=0.7)

    assert confidence == 0.48
    assert 0.0 <= score <= 1.0
    assert score > 0.5
