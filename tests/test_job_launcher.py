from __future__ import annotations

import json
from pathlib import Path
from time import sleep

from image_preference_modelling.jobs.job_launcher import JobLauncher
from image_preference_modelling.storage.state_store import StateStore


def _wait_for_terminal_state(
    store: StateStore, run_id: str, timeout_seconds: float = 2.0
) -> dict[str, str | None]:
    deadline = timeout_seconds
    elapsed = 0.0
    while elapsed < deadline:
        run = store.get_run(run_id)
        assert run is not None
        if run["status"] in {"completed", "failed", "cancelled"}:
            return run
        sleep(0.02)
        elapsed += 0.02
    run = store.get_run(run_id)
    assert run is not None
    return run


def _wait_for_log_file(run: dict[str, str | None], timeout_seconds: float = 2.0) -> Path:
    artifact_dir = run.get("artifact_dir")
    assert artifact_dir is not None
    log_file = Path(artifact_dir) / "job.log"
    deadline = timeout_seconds
    elapsed = 0.0
    while elapsed < deadline:
        if log_file.exists():
            return log_file
        sleep(0.02)
        elapsed += 0.02
    return log_file


def test_start_run_marks_run_completed_and_writes_log(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    launcher = JobLauncher(state_store=store)

    run_id = store.create_run(
        run_type="reward_model",
        display_name="Train PickScore_me v0",
        config={"pairs": 240},
    )
    launcher.start_run(run_id)

    run = _wait_for_terminal_state(store, run_id)
    assert run["status"] == "completed"
    assert run["finished_at"] is not None
    log_file = _wait_for_log_file(run)
    assert log_file.exists()
    assert "COMPLETED" in log_file.read_text(encoding="utf-8")


def test_dispatch_run_failure_path_sets_failed_and_logs(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    launcher = JobLauncher(state_store=store)
    run_id = store.create_run(
        run_type="evaluation",
        display_name="Failing run",
        config={"force_fail": True},
    )

    launcher.dispatch_run(run_id)
    run = _wait_for_terminal_state(store, run_id)
    assert run["status"] == "failed"
    log_text = _wait_for_log_file(run).read_text(encoding="utf-8")
    assert "forced failure" in log_text


def test_cancel_running_run_transitions_to_cancelled(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    launcher = JobLauncher(state_store=store)
    run_id = store.create_run(
        run_type="generation",
        display_name="Cancellable run",
        config={"simulated_steps": 20, "simulated_step_seconds": 0.01},
    )

    launcher.dispatch_run(run_id)
    sleep(0.04)
    launcher.cancel_run(run_id)

    run = _wait_for_terminal_state(store, run_id)
    assert run["status"] == "cancelled"


def test_gepa_run_config_preserves_job_id_and_minibatch_size(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    launcher = JobLauncher(state_store=store)
    job_id = store.create_aesthetic_job(
        name="launcher-job",
        description="job for launcher test",
        seed_system_prompt="Preserve intent while improving style.",
    )
    session_id = store.create_rating_session(name="launcher-session")
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text="city at dusk",
        intent_text="city at dusk",
        baseline_image_uri="baseline.png",
        candidate_image_uri="candidate.png",
        candidate_id=None,
        system_prompt="Improve color depth.",
        generation_mode="text_only",
        model_config={"image_model": "test-model"},
    )
    comparison_id = store.add_comparison(
        session_id=session_id,
        prompt_text="city at dusk",
        left_image_uri="baseline.png",
        right_image_uri="candidate.png",
        winner="right",
        critique="Refined output has stronger contrast and detail.",
        outcome="winner",
    )
    store.mark_rollout_feedback_complete(rollout_id, comparison_id)

    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA optimization",
        config={
            "job_id": job_id,
            "minibatch_size": 1,
            "selected_rollout_ids": [rollout_id],
            "optimizer_backend": "heuristic_fallback",
        },
    )

    launcher.dispatch_run(run_id)
    run = _wait_for_terminal_state(store, run_id)
    assert run["status"] == "completed"

    config = json.loads((Path(run["artifact_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert config["job_id"] == job_id
    assert config["minibatch_size"] == 1
    checkpoint_path = Path(run["artifact_dir"]) / "checkpoint.json"
    assert checkpoint_path.exists()
