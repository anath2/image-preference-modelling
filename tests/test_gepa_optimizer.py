from __future__ import annotations

import json
from pathlib import Path

from image_preference_modelling.gepa.optimizer import run_gepa_optimization
from image_preference_modelling.storage.state_store import StateStore


def _create_completed_rollout(
    store: StateStore,
    *,
    job_id: str,
    prompt: str,
    winner: str | None,
    outcome: str,
    critique: str,
) -> str:
    session_id = store.create_rating_session(name=f"session-{prompt}")
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text=prompt,
        intent_text=prompt,
        baseline_image_uri=f"{prompt}-baseline.png",
        candidate_image_uri=f"{prompt}-candidate.png",
        candidate_id=None,
        system_prompt="Refine carefully.",
        generation_mode="text_only",
        model_config={"image_model": "test-model"},
    )
    comparison_id = store.add_comparison(
        session_id=session_id,
        prompt_text=prompt,
        left_image_uri=f"{prompt}-baseline.png",
        right_image_uri=f"{prompt}-candidate.png",
        winner=winner,
        critique=critique,
        outcome=outcome,  # type: ignore[arg-type]
    )
    store.mark_rollout_feedback_complete(rollout_id, comparison_id)
    return rollout_id


def test_run_gepa_optimization_writes_checkpoint_and_promotes_candidate(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="cinematic",
        description="Cinematic polish",
        seed_system_prompt="Improve cinematic depth while preserving layout.",
    )
    rollout_a = _create_completed_rollout(
        store,
        job_id=job_id,
        prompt="p1",
        winner="right",
        outcome="winner",
        critique="Refined output is stronger with better lighting and depth.",
    )
    rollout_b = _create_completed_rollout(
        store,
        job_id=job_id,
        prompt="p2",
        winner=None,
        outcome="both_good",
        critique="Both are good, refined has richer atmosphere.",
    )
    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA run",
        config={
            "job_id": job_id,
            "minibatch_size": 2,
            "selected_rollout_ids": [rollout_a, rollout_b],
            "active_candidate_id": None,
            "compiled_prompt": None,
        },
    )
    run = store.get_run(run_id)
    assert run is not None

    events: list[tuple[str, str]] = []
    run_gepa_optimization(
        run_id=run_id,
        artifact_dir=Path(run["artifact_dir"]),
        state_store=store,
        config=json.loads((Path(run["artifact_dir"]) / "config.json").read_text(encoding="utf-8")),
        append_event=lambda level, message: events.append((level, message)),
        is_cancel_requested=lambda: False,
    )

    checkpoint_path = Path(run["artifact_dir"]) / "checkpoint.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["job_id"] == job_id
    assert checkpoint["minibatch_size"] == 2
    assert set(checkpoint["selected_rollout_ids"]) == {rollout_a, rollout_b}
    assert "new_candidate_id" in checkpoint
    assert checkpoint["optimizer_backend"] in {"dspy_gepa", "heuristic_fallback"}

    job = store.get_aesthetic_job(job_id)
    assert job is not None
    assert job["active_candidate_id"] == checkpoint["new_candidate_id"]
    assert job["compiled_system_prompt"] == checkpoint["compiled_prompt"]
    assert job["baseline_system_prompt"] == "Improve cinematic depth while preserving layout."
    assert job["latest_system_prompt"] == checkpoint["compiled_prompt"]


def test_run_gepa_optimization_falls_back_when_dspy_not_configured(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DSPY_MODEL", raising=False)
    monkeypatch.delenv("PROMPT_MODEL", raising=False)
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="fallback-job",
        description="Fallback behavior check",
        seed_system_prompt="Keep composition while improving texture.",
    )
    rollout = _create_completed_rollout(
        store,
        job_id=job_id,
        prompt="p1",
        winner="right",
        outcome="winner",
        critique="Refined version has clearer foreground detail.",
    )
    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA fallback run",
        config={
            "job_id": job_id,
            "minibatch_size": 1,
            "selected_rollout_ids": [rollout],
            "optimizer_backend": "dspy_gepa",
            "dspy_model": "",
            "openrouter_api_key": "",
        },
    )
    run = store.get_run(run_id)
    assert run is not None
    run_gepa_optimization(
        run_id=run_id,
        artifact_dir=Path(run["artifact_dir"]),
        state_store=store,
        config=json.loads((Path(run["artifact_dir"]) / "config.json").read_text(encoding="utf-8")),
        append_event=lambda _level, _message: None,
        is_cancel_requested=lambda: False,
    )
    checkpoint = json.loads((Path(run["artifact_dir"]) / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["optimizer_backend"] == "heuristic_fallback"


def test_run_gepa_optimization_uses_frontier_parent_candidate(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="frontier-parent",
        description="Frontier parent selection check",
        seed_system_prompt="Seed policy.",
    )
    rollout = _create_completed_rollout(
        store,
        job_id=job_id,
        prompt="p1",
        winner="right",
        outcome="winner",
        critique="Refined version has better atmosphere.",
    )
    seed_run_id = store.create_run(
        run_type="gepa",
        display_name="Existing candidate run",
        config={"job_id": job_id, "minibatch_size": 1, "selected_rollout_ids": []},
    )
    parent_candidate_id = store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=[],
        candidate_text="Parent policy",
        compiled_prompt="Frontier parent policy.",
        objective_scores={"preference_win": 0.8, "feedback_quality": 0.7},
        created_by_run_id=seed_run_id,
    )
    store.set_candidate_frontier_membership(parent_candidate_id, True)
    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA frontier-parent run",
        config={
            "job_id": job_id,
            "minibatch_size": 1,
            "selected_rollout_ids": [rollout],
            "optimizer_backend": "heuristic",
            "candidate_selection_seed": 0,
        },
    )
    run = store.get_run(run_id)
    assert run is not None

    run_gepa_optimization(
        run_id=run_id,
        artifact_dir=Path(run["artifact_dir"]),
        state_store=store,
        config=json.loads((Path(run["artifact_dir"]) / "config.json").read_text(encoding="utf-8")),
        append_event=lambda _level, _message: None,
        is_cancel_requested=lambda: False,
    )

    checkpoint = json.loads((Path(run["artifact_dir"]) / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["parent_candidate_id"] == parent_candidate_id
    assert checkpoint["compiled_prompt"].startswith("Frontier parent policy.")
    assert any(item["candidate_id"] == checkpoint["new_candidate_id"] for item in checkpoint["frontier_snapshot"])
