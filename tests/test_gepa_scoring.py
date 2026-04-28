from image_preference_modelling.gepa.scoring import score_rollout_feedback


def _rollout(outcome: str, winner: str | None, critique: str) -> dict[str, str | None]:
    return {"outcome": outcome, "winner": winner, "critique": critique}


def test_score_rollout_feedback_candidate_win() -> None:
    scores = score_rollout_feedback(
        _rollout("winner", "right", "Great lighting and composition improvements.")
    )
    assert scores["preference_win"] == 1.0
    assert scores["feedback_quality"] == 1.0


def test_score_rollout_feedback_baseline_win() -> None:
    scores = score_rollout_feedback(_rollout("winner", "left", "Baseline keeps stronger framing."))
    assert scores["preference_win"] == 0.0


def test_score_rollout_feedback_both_good() -> None:
    scores = score_rollout_feedback(_rollout("both_good", None, "Both are good but different vibes."))
    assert scores["preference_win"] == 0.5


def test_score_rollout_feedback_both_bad() -> None:
    scores = score_rollout_feedback(_rollout("both_bad", None, "Both miss the intended mood."))
    assert scores["preference_win"] == 0.5


def test_score_rollout_feedback_cant_decide_short_critique() -> None:
    scores = score_rollout_feedback(_rollout("cant_decide", None, "close"))
    assert scores["preference_win"] == 0.5
    assert scores["feedback_quality"] == 0.5
    assert scores["intent_preservation"] == 1.0
    assert scores["composition_preservation"] == 1.0
