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
    assert store.schema_version() == 8

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


def test_migrate_v4_rollouts_adds_text_only_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta(key, value) VALUES('schema_version', '4');
            CREATE TABLE aesthetic_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                seed_refinement_prompt TEXT NOT NULL,
                active_candidate_id TEXT,
                compiled_gepa_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE rollouts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                comparison_id TEXT,
                prompt_text TEXT NOT NULL,
                intent_text TEXT NOT NULL,
                baseline_image_uri TEXT NOT NULL,
                refined_image_uri TEXT NOT NULL,
                candidate_id TEXT,
                refinement_prompt TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                feedback_completed_at TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO aesthetic_jobs(
                id, name, description, status, seed_refinement_prompt,
                active_candidate_id, compiled_gepa_prompt, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job_legacy",
                "legacy-job",
                "legacy",
                "active",
                "legacy seed",
                None,
                "legacy compiled",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO rollouts(
                id, job_id, comparison_id, prompt_text, intent_text, baseline_image_uri,
                refined_image_uri, candidate_id, refinement_prompt, model_config_json,
                status, created_at, feedback_completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rollout_legacy",
                "job_legacy",
                None,
                "prompt",
                "intent",
                "baseline.png",
                "refined.png",
                None,
                "legacy refinement",
                "{\"model\":\"legacy\"}",
                "generated",
                "2026-01-01T00:00:00+00:00",
                None,
            ),
        )
        connection.commit()

    store = StateStore(db_path=db_path, artifact_root=artifact_root)
    assert store.schema_version() == 8

    job = store.get_aesthetic_job("job_legacy")
    assert job is not None
    assert job["seed_system_prompt"] == "legacy seed"
    assert job["compiled_system_prompt"] == "legacy compiled"

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT candidate_image_uri, system_prompt, generation_mode, rollout_type,
                   right_system_prompt_snapshot
            FROM rollouts
            WHERE id = ?
            """,
            ("rollout_legacy",),
        ).fetchone()
    assert row is not None
    assert row[0] == "refined.png"
    assert row[1] == "legacy refinement"
    assert row[2] == "image_conditioned"
    assert row[3] == "baseline_candidate"
    assert row[4] == "legacy refinement"


def test_aesthetic_job_crud_and_policy_update(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="cinematic-neon",
        description="Neon cinematic mood with preserved composition",
        seed_system_prompt="Improve lighting and texture while preserving composition.",
    )

    listed = store.list_aesthetic_jobs()
    assert len(listed) == 1
    assert listed[0]["id"] == job_id
    job = store.get_aesthetic_job(job_id)
    assert job is not None
    assert job["status"] == "active"

    store.update_aesthetic_job_policy(
        job_id, active_candidate_id="candidate_001", compiled_system_prompt="Prefer richer neon contrast."
    )
    updated = store.get_aesthetic_job(job_id)
    assert updated is not None
    assert updated["active_candidate_id"] == "candidate_001"
    assert updated["compiled_system_prompt"] == "Prefer richer neon contrast."
    assert updated["baseline_system_prompt"] == ""
    assert updated["latest_system_prompt"] == "Improve lighting and texture while preserving composition."


def test_create_rollout_and_mark_feedback_complete(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="portrait-polish",
        description="Polish portrait tone and detail",
        seed_system_prompt="Improve facial detail while preserving identity.",
    )
    session_id = store.create_rating_session(name="session")
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text="portrait in golden hour",
        intent_text="portrait in warm natural light",
        baseline_image_uri="baseline.png",
        candidate_image_uri="candidate.png",
        candidate_id=None,
        system_prompt="Enhance natural warm tones.",
        generation_mode="text_only",
        model_config={"model": "image-model-a"},
    )

    comparison_id = store.add_comparison(
        session_id=session_id,
        prompt_text="portrait in golden hour",
        left_image_uri="baseline.png",
        right_image_uri="candidate.png",
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
    assert completed[0]["candidate_image_uri"] == "candidate.png"
    assert completed[0]["system_prompt"] == "Enhance natural warm tones."
    assert completed[0]["rollout_type"] == "baseline_candidate"
    assert completed[0]["left_candidate_id"] is None
    assert completed[0]["right_candidate_id"] is None
    assert completed[0]["left_system_prompt_snapshot"] == ""
    assert completed[0]["right_system_prompt_snapshot"] == "Enhance natural warm tones."
    assert completed[0]["selection_mode"] is None
    assert completed[0]["llm_score"] is None
    assert completed[0]["llm_reason"] is None
    assert completed[0]["generation_mode"] == "text_only"
    assert store.count_completed_rollouts_for_job(job_id) == 1
    assert store.list_completed_rollout_ids_for_job(job_id, limit=3) == [rollout_id]


def test_create_candidate_comparison_rollout_metadata(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="candidate-pool",
        description="candidate pool checks",
        seed_system_prompt="seed",
    )
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text="prompt",
        intent_text="intent",
        baseline_image_uri="left.png",
        candidate_image_uri="right.png",
        candidate_id="candidate_right",
        system_prompt="right policy",
        rollout_type="candidate_comparison",
        left_candidate_id="candidate_left",
        right_candidate_id="candidate_right",
        left_system_prompt_snapshot="left policy",
        right_system_prompt_snapshot="right policy",
        generation_mode="text_only",
        model_config={"model": "image-model-a"},
    )

    rollout = store.list_rollouts_for_job(job_id)[0]
    assert rollout["id"] == rollout_id
    assert rollout["rollout_type"] == "candidate_comparison"
    assert rollout["left_candidate_id"] == "candidate_left"
    assert rollout["right_candidate_id"] == "candidate_right"
    assert rollout["left_system_prompt_snapshot"] == "left policy"
    assert rollout["right_system_prompt_snapshot"] == "right policy"


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


def test_gepa_candidate_creation_listing_and_promotion(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="mood-noir",
        description="Noir mood enhancement",
        seed_system_prompt="Increase contrast while preserving scene composition.",
    )
    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA seed run",
        config={"job_id": job_id, "minibatch_size": 1, "selected_rollout_ids": []},
    )

    candidate_id = store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=[],
        candidate_text="Candidate policy text",
        compiled_prompt="Use moody shadows and controlled highlights.",
        objective_scores={"preference_win": 0.8, "feedback_quality": 0.9},
        created_by_run_id=run_id,
    )
    candidates = store.list_gepa_candidates_for_job(job_id)
    assert len(candidates) == 1
    assert candidates[0]["id"] == candidate_id
    assert candidates[0]["parent_candidate_ids"] == []
    assert candidates[0]["objective_scores"]["preference_win"] == 0.8
    assert candidates[0]["frontier_member"] is False

    store.set_candidate_frontier_membership(candidate_id, True)
    updated_candidates = store.list_gepa_candidates_for_job(job_id)
    assert updated_candidates[0]["frontier_member"] is True

    store.promote_job_candidate(job_id, candidate_id)
    job = store.get_aesthetic_job(job_id)
    assert job is not None
    assert job["active_candidate_id"] == candidate_id
    assert job["compiled_system_prompt"] == "Use moody shadows and controlled highlights."
    assert job["baseline_system_prompt"] == "Increase contrast while preserving scene composition."
    assert job["latest_system_prompt"] == "Use moody shadows and controlled highlights."


def test_recompute_gepa_frontier_marks_dominated_candidates(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="frontier",
        description="frontier checks",
        seed_system_prompt="seed",
    )
    run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA frontier run",
        config={"job_id": job_id, "minibatch_size": 1, "selected_rollout_ids": []},
    )
    weak_candidate = store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=[],
        candidate_text="weak",
        compiled_prompt="weak",
        objective_scores={"preference_win": 0.2, "feedback_quality": 0.2},
        created_by_run_id=run_id,
    )
    strong_candidate = store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=[],
        candidate_text="strong",
        compiled_prompt="strong",
        objective_scores={"preference_win": 0.8, "feedback_quality": 0.8},
        created_by_run_id=run_id,
    )

    snapshot = store.recompute_gepa_frontier_for_job(job_id)
    membership = {item["candidate_id"]: item["frontier_member"] for item in snapshot}
    assert membership[weak_candidate] is False
    assert membership[strong_candidate] is True

    candidates = {item["id"]: item for item in store.list_gepa_candidates_for_job(job_id)}
    assert candidates[weak_candidate]["frontier_member"] is False
    assert candidates[strong_candidate]["frontier_member"] is True


def test_update_archive_gate_and_rollover_job_state(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="portrait",
        description="portrait style",
        seed_system_prompt="seed prompt",
        sampling_profile={"category": "portrait"},
        gepa_enable_threshold=2,
    )
    store.update_aesthetic_job(
        job_id,
        name="portrait-v2",
        description="portrait style v2",
        sampling_profile={"category": "outdoor_landscape"},
        gepa_enable_threshold=3,
    )
    updated = store.get_aesthetic_job(job_id)
    assert updated is not None
    assert updated["name"] == "portrait-v2"
    assert updated["description"] == "portrait style v2"
    assert updated["gepa_enable_threshold"] == 3

    session_id = store.create_rating_session(name="gate-session")
    rollout_a = store.create_rollout(
        job_id=job_id,
        prompt_text="p1",
        intent_text="p1",
        baseline_image_uri="baseline-1.png",
        candidate_image_uri="candidate-1.png",
        candidate_id=None,
        system_prompt="latest-1",
        baseline_system_prompt_snapshot="",
        latest_system_prompt_snapshot="seed prompt",
        prompt_category="outdoor_landscape",
        selection_mode="llm_guided",
        llm_score=0.88,
        llm_reason="Strong match",
        generation_mode="text_only",
        model_config={"m": 1},
    )
    cmp_a = store.add_comparison(
        session_id=session_id,
        prompt_text="p1",
        left_image_uri="baseline-1.png",
        right_image_uri="candidate-1.png",
        winner="left",
        critique="baseline better",
        outcome="winner",
    )
    store.mark_rollout_feedback_complete(rollout_a, cmp_a)
    rollout_b = store.create_rollout(
        job_id=job_id,
        prompt_text="p2",
        intent_text="p2",
        baseline_image_uri="baseline-2.png",
        candidate_image_uri="candidate-2.png",
        candidate_id=None,
        system_prompt="latest-1",
        baseline_system_prompt_snapshot="",
        latest_system_prompt_snapshot="seed prompt",
        prompt_category="outdoor_landscape",
        selection_mode="keyword_fallback",
        llm_score=None,
        llm_reason=None,
        generation_mode="text_only",
        model_config={"m": 1},
    )
    cmp_b = store.add_comparison(
        session_id=session_id,
        prompt_text="p2",
        left_image_uri="baseline-2.png",
        right_image_uri="candidate-2.png",
        winner="left",
        critique="baseline better again",
        outcome="winner",
    )
    store.mark_rollout_feedback_complete(rollout_b, cmp_b)
    gate = store.get_gepa_gate_status(job_id)
    assert gate["completed_feedback_count"] == 2
    assert gate["new_feedback_count"] == 2
    assert gate["threshold"] == 3
    assert gate["enabled"] is False
    assert gate["last_gepa_completed_at"] is None
    assert set(store.list_gepa_eligible_rollout_ids_for_job(job_id, limit=3)) == {
        rollout_a,
        rollout_b,
    }

    gepa_run_id = store.create_run(
        run_type="gepa",
        display_name="GEPA gate checkpoint",
        config={"job_id": job_id, "minibatch_size": 2, "selected_rollout_ids": [rollout_a, rollout_b]},
    )
    store.update_run_status(gepa_run_id, "running")
    store.update_run_status(gepa_run_id, "completed")
    gate_after_run = store.get_gepa_gate_status(job_id)
    assert gate_after_run["completed_feedback_count"] == 2
    assert gate_after_run["new_feedback_count"] == 0
    assert gate_after_run["enabled"] is False
    assert gate_after_run["last_gepa_completed_at"] is not None
    assert store.list_gepa_eligible_rollout_ids_for_job(job_id, limit=3) == []

    store.rollover_job_system_prompt(job_id, latest_system_prompt="optimized prompt")
    rolled = store.get_aesthetic_job(job_id)
    assert rolled is not None
    assert rolled["baseline_system_prompt"] == "seed prompt"
    assert rolled["latest_system_prompt"] == "optimized prompt"

    store.archive_aesthetic_job(job_id)
    assert store.get_aesthetic_job(job_id)["status"] == "archived"  # type: ignore[index]


def test_list_rollouts_for_job_returns_metadata_and_feedback(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    job_id = store.create_aesthetic_job(
        name="inspector",
        description="inspector job",
        seed_system_prompt="seed",
    )
    session_id = store.create_rating_session(name="inspector-session")
    rollout_id = store.create_rollout(
        job_id=job_id,
        prompt_text="prompt",
        intent_text="intent",
        baseline_image_uri="baseline.png",
        candidate_image_uri="candidate.png",
        candidate_id=None,
        system_prompt="system prompt",
        prompt_category="portrait",
        selection_mode="llm_guided",
        llm_score=0.93,
        llm_reason="strong fit",
        generation_mode="text_only",
        model_config={"image_model": "test"},
    )
    comparison_id = store.add_comparison(
        session_id=session_id,
        prompt_text="prompt",
        left_image_uri="baseline.png",
        right_image_uri="candidate.png",
        winner="right",
        critique="candidate better",
        outcome="winner",
    )
    store.mark_rollout_feedback_complete(rollout_id, comparison_id)

    rollouts = store.list_rollouts_for_job(job_id)
    assert len(rollouts) == 1
    item = rollouts[0]
    assert item["id"] == rollout_id
    assert item["selection_mode"] == "llm_guided"
    assert item["llm_score"] == 0.93
    assert item["llm_reason"] == "strong fit"
    assert item["winner"] == "right"
    assert item["outcome"] == "winner"
    assert item["model_config"]["image_model"] == "test"
