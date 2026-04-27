from pathlib import Path
import sqlite3

import pytest

from image_preference_modelling.storage.state_store import StateStore


def test_create_run_persists_artifact_paths(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")

    run_id = store.create_run(
        run_type="generation",
        display_name="Bootstrap generation batch",
        config={"prompt_set": "bootstrap-v1", "seeds": [11, 22, 33, 44]},
    )

    run = store.get_run(run_id)
    assert run is not None
    assert run["run_type"] == "generation"
    assert run["status"] == "queued"
    assert Path(run["artifact_dir"]).exists()
    assert (Path(run["artifact_dir"]) / "config.json").exists()
    assert Path(run["artifact_dir"]).parent.name == "generation_runs"

    with sqlite3.connect(tmp_path / "state.db") as connection:
        typed_row = connection.execute(
            "SELECT run_id, seed_count FROM generation_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    assert typed_row is not None
    assert typed_row[0] == run_id
    assert typed_row[1] == 4


def test_create_rating_session_and_append_comparison(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    session_id = store.create_rating_session(name="bootstrap-session")

    store.add_comparison(
        session_id=session_id,
        prompt_text="a retro sci-fi street scene",
        left_image_uri="artifacts/l.png",
        right_image_uri="artifacts/r.png",
        winner="left",
        critique="Left has better composition and cleaner light.",
        outcome="winner",
    )

    rows = store.list_recent_comparisons(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["rating_session_id"] == session_id
    assert row["winner"] == "left"
    assert row["outcome"] == "winner"
    session = store.get_rating_session(session_id)
    assert session is not None
    assert Path(session["artifact_dir"]).exists()
    assert Path(session["artifact_dir"]).parent.name == "rating_sessions"


def test_run_status_transitions_and_events(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    run_id = store.create_run(
        run_type="generation",
        display_name="Transition checks",
        config={"note": "transition-test"},
    )

    store.update_run_status(run_id, "running")
    store.update_run_status(run_id, "completed")

    run = store.get_run(run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["started_at"] is not None
    assert run["finished_at"] is not None

    events = store.list_run_events(run_id)
    assert len(events) >= 1
    assert events[0]["message"] == "Run queued for dispatch."


def test_invalid_transition_raises_error(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    run_id = store.create_run(
        run_type="evaluation",
        display_name="Invalid transition run",
        config={"note": "invalid-transition"},
    )
    store.update_run_status(run_id, "running")
    store.update_run_status(run_id, "completed")

    with pytest.raises(ValueError, match="Invalid transition"):
        store.update_run_status(run_id, "running")


def test_typed_tables_created_for_non_generation_runs(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    reward_run = store.create_run(
        run_type="reward_model",
        display_name="Reward v1",
        config={"version_name": "reward-v1"},
    )
    gepa_run = store.create_run(
        run_type="gepa",
        display_name="GEPA loop 1",
        config={"parent_reward_model_run_id": reward_run},
    )
    eval_run = store.create_run(
        run_type="evaluation",
        display_name="Eval loop 1",
        config={"baseline_run_id": reward_run, "candidate_run_id": gepa_run},
    )

    with sqlite3.connect(tmp_path / "state.db") as connection:
        reward_row = connection.execute(
            "SELECT run_id FROM reward_model_versions WHERE run_id = ?",
            (reward_run,),
        ).fetchone()
        gepa_row = connection.execute(
            "SELECT run_id FROM gepa_runs WHERE run_id = ?",
            (gepa_run,),
        ).fetchone()
        eval_row = connection.execute(
            "SELECT run_id FROM evaluation_runs WHERE run_id = ?",
            (eval_run,),
        ).fetchone()

    assert reward_row is not None
    assert gepa_row is not None
    assert eval_row is not None
    reward = store.get_run(reward_run)
    gepa = store.get_run(gepa_run)
    evaluation = store.get_run(eval_run)
    assert reward is not None
    assert gepa is not None
    assert evaluation is not None
    assert Path(reward["artifact_dir"]).parent.name == "reward_model_versions"
    assert Path(gepa["artifact_dir"]).parent.name == "gepa_runs"
    assert Path(evaluation["artifact_dir"]).parent.name == "evaluation_runs"


def test_schema_version_and_migration_from_v1(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                run_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );
            CREATE TABLE rating_sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE comparisons (
                id TEXT PRIMARY KEY,
                rating_session_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                left_image_uri TEXT NOT NULL,
                right_image_uri TEXT NOT NULL,
                winner TEXT,
                critique TEXT,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO rating_sessions(id, name, status, created_at)
            VALUES ('session_legacy', 'legacy', 'active', '2026-01-01T00:00:00+00:00');
            """
        )
        connection.commit()

    store = StateStore(db_path=db_path, artifact_root=artifact_root)
    assert store.schema_version() == 3

    session = store.get_rating_session("session_legacy")
    assert session is not None
    assert session["artifact_dir"].endswith("/rating_sessions/session_legacy")


def test_integrity_report_detects_bad_rows(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    run_id = store.create_run(
        run_type="generation",
        display_name="Corrupt me",
        config={"note": "integrity"},
    )

    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute(
            """
            UPDATE runs
            SET status = ?, started_at = NULL, finished_at = ?
            WHERE id = ?
            """,
            ("running", "2026-01-01T00:00:00+00:00", run_id),
        )
        connection.execute(
            "UPDATE runs SET artifact_dir = ? WHERE id = ?",
            (str(tmp_path / "missing" / run_id), run_id),
        )
        connection.execute(
            """
            INSERT INTO comparisons(
                id, rating_session_id, prompt_text, left_image_uri, right_image_uri,
                winner, critique, outcome, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cmp_bad",
                "missing_session",
                "p",
                "l",
                "r",
                None,
                "bad",
                "winner",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()

    issues = store.integrity_report()
    assert any("running status without started_at" in msg for msg in issues["invalid_status_transitions"])
    assert any("missing artifact directory" in msg for msg in issues["dangling_artifacts"])
    assert any("missing rating_session" in msg for msg in issues["invalid_comparisons"])


def test_aesthetic_job_crud_and_policy_update(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="cinematic-neon",
        description="Neon cinematic mood with preserved composition",
        seed_refinement_prompt="Improve lighting and texture while preserving composition.",
    )

    listed = store.list_aesthetic_jobs()
    assert len(listed) == 1
    assert listed[0]["id"] == job_id
    job = store.get_aesthetic_job(job_id)
    assert job is not None
    assert job["status"] == "active"

    store.update_aesthetic_job_policy(
        job_id, active_candidate_id="candidate_001", compiled_gepa_prompt="Prefer richer neon contrast."
    )
    updated = store.get_aesthetic_job(job_id)
    assert updated is not None
    assert updated["active_candidate_id"] == "candidate_001"
    assert updated["compiled_gepa_prompt"] == "Prefer richer neon contrast."


def test_create_rollout_and_mark_feedback_complete(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="portrait-polish",
        description="Polish portrait tone and detail",
        seed_refinement_prompt="Improve facial detail while preserving identity.",
    )
    session_id = store.create_rating_session(name="session")
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text="portrait in golden hour",
        intent_text="portrait in warm natural light",
        baseline_image_uri="baseline.png",
        refined_image_uri="refined.png",
        candidate_id=None,
        refinement_prompt="Enhance natural warm tones.",
        model_config={"model": "image-model-a"},
    )

    comparison_id = store.add_comparison(
        session_id=session_id,
        prompt_text="portrait in golden hour",
        left_image_uri="baseline.png",
        right_image_uri="refined.png",
        winner="right",
        critique="Refined image has better skin tone and lighting consistency.",
        outcome="winner",
    )
    store.mark_rollout_feedback_complete(rollout_id, comparison_id)

    completed = store.list_completed_rollouts_for_job(job_id)
    assert len(completed) == 1
    assert completed[0]["id"] == rollout_id
    assert completed[0]["status"] == "feedback_complete"
    assert completed[0]["comparison_id"] == comparison_id


def test_add_comparison_requires_non_empty_critique(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    session_id = store.create_rating_session(name="strict-feedback")

    with pytest.raises(ValueError, match="critique cannot be empty"):
        store.add_comparison(
            session_id=session_id,
            prompt_text="prompt",
            left_image_uri="left.png",
            right_image_uri="right.png",
            winner="left",
            critique="   ",
            outcome="winner",
        )
