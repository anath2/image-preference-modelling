from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_preference_modelling.storage.contracts import (
    RatingOutcome,
    RunStatus,
    RunType,
    rating_session_artifact_dir,
    run_artifact_dir,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_VERSION = 5
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
VALID_RUN_STATUSES = {"queued", "running", *TERMINAL_RUN_STATUSES}
VALID_RUN_TYPES = {"generation", "reward_model", "gepa", "evaluation"}
VALID_RATING_SESSION_STATUSES = {"active", "archived"}
VALID_RATING_OUTCOMES = {"winner", "both_good", "both_bad", "cant_decide"}
VALID_AESTHETIC_JOB_STATUSES = {"active", "archived"}
VALID_ROLLOUT_STATUSES = {"generated", "feedback_complete"}
VALID_ROLLOUT_GENERATION_MODES = {"image_conditioned", "text_only"}


class StateStore:
    """Durable state for runs, sessions, and human feedback."""

    def __init__(self, db_path: Path, artifact_root: Path) -> None:
        self.db_path = db_path
        self.artifact_root = artifact_root
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            self._create_schema_v5(connection)
            self._migrate_to_latest(connection)
            connection.commit()

    def _schema_version(self, connection: sqlite3.Connection) -> int:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'"
        ).fetchone()
        if not table_exists:
            return 0
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return 0

    def _set_schema_version(self, connection: sqlite3.Connection, version: int) -> None:
        connection.execute(
            """
            INSERT INTO schema_meta(key, value)
            VALUES('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (str(version),),
        )

    def _create_schema_v5(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
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

            CREATE TABLE IF NOT EXISTS rating_sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS comparisons (
                id TEXT PRIMARY KEY,
                rating_session_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                left_image_uri TEXT NOT NULL,
                right_image_uri TEXT NOT NULL,
                winner TEXT,
                critique TEXT,
                outcome TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(rating_session_id) REFERENCES rating_sessions(id)
            );

            CREATE TABLE IF NOT EXISTS run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS prompt_sets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generation_runs (
                run_id TEXT PRIMARY KEY,
                prompt_set_id TEXT,
                model_name TEXT,
                seed_count INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                FOREIGN KEY(prompt_set_id) REFERENCES prompt_sets(id)
            );

            CREATE TABLE IF NOT EXISTS reward_model_versions (
                run_id TEXT PRIMARY KEY,
                version_name TEXT NOT NULL,
                base_model TEXT,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS gepa_runs (
                run_id TEXT PRIMARY KEY,
                parent_reward_model_run_id TEXT,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_reward_model_run_id) REFERENCES reward_model_versions(run_id)
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                baseline_run_id TEXT,
                candidate_run_id TEXT,
                status TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
                FOREIGN KEY(baseline_run_id) REFERENCES runs(id),
                FOREIGN KEY(candidate_run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS aesthetic_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                seed_system_prompt TEXT NOT NULL,
                active_candidate_id TEXT,
                compiled_system_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rollouts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                comparison_id TEXT,
                prompt_text TEXT NOT NULL,
                intent_text TEXT NOT NULL,
                baseline_image_uri TEXT NOT NULL,
                candidate_image_uri TEXT NOT NULL,
                candidate_id TEXT,
                system_prompt TEXT NOT NULL,
                generation_mode TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                feedback_completed_at TEXT,
                FOREIGN KEY(job_id) REFERENCES aesthetic_jobs(id),
                FOREIGN KEY(comparison_id) REFERENCES comparisons(id)
            );

            CREATE TABLE IF NOT EXISTS gepa_candidates (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                parent_candidate_ids_json TEXT NOT NULL,
                candidate_text TEXT NOT NULL,
                compiled_prompt TEXT NOT NULL,
                objective_scores_json TEXT NOT NULL,
                frontier_member INTEGER NOT NULL DEFAULT 0,
                created_by_run_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES aesthetic_jobs(id),
                FOREIGN KEY(created_by_run_id) REFERENCES runs(id)
            );
            """
        )

    def _migrate_to_latest(self, connection: sqlite3.Connection) -> None:
        current_version = self._schema_version(connection)

        if current_version == 0:
            self._migrate_v0_to_v1(connection)
            current_version = 1

        if current_version == 1:
            self._migrate_v1_to_v2(connection)
            current_version = 2

        if current_version == 2:
            self._migrate_v2_to_v3(connection)
            current_version = 3

        if current_version == 3:
            self._migrate_v3_to_v4(connection)
            current_version = 4

        if current_version == 4:
            self._migrate_v4_to_v5(connection)
            current_version = 5

        if current_version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported schema version {current_version}; expected {SCHEMA_VERSION}"
            )

    def _migrate_v0_to_v1(self, connection: sqlite3.Connection) -> None:
        # v0 had no schema metadata. Treat existing table layout as v1-compatible.
        self._set_schema_version(connection, 1)

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(rating_sessions)").fetchall()
        }
        if "artifact_dir" not in session_columns:
            connection.execute(
                "ALTER TABLE rating_sessions ADD COLUMN artifact_dir TEXT NOT NULL DEFAULT ''"
            )

        # Backfill canonical artifact dirs without rewriting existing run artifact_dir values.
        rows = connection.execute("SELECT id FROM rating_sessions").fetchall()
        for row in rows:
            session_id = row[0]
            artifact_dir = str(rating_session_artifact_dir(self.artifact_root, session_id))
            connection.execute(
                "UPDATE rating_sessions SET artifact_dir = ? WHERE id = ?",
                (artifact_dir, session_id),
            )

        self._set_schema_version(connection, 2)

    def _migrate_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS aesthetic_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                seed_refinement_prompt TEXT NOT NULL,
                active_candidate_id TEXT,
                compiled_gepa_prompt TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rollouts (
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
                feedback_completed_at TEXT,
                FOREIGN KEY(job_id) REFERENCES aesthetic_jobs(id),
                FOREIGN KEY(comparison_id) REFERENCES comparisons(id)
            )
            """
        )
        self._set_schema_version(connection, 3)

    def _migrate_v3_to_v4(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS gepa_candidates (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                parent_candidate_ids_json TEXT NOT NULL,
                candidate_text TEXT NOT NULL,
                compiled_prompt TEXT NOT NULL,
                objective_scores_json TEXT NOT NULL,
                frontier_member INTEGER NOT NULL DEFAULT 0,
                created_by_run_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES aesthetic_jobs(id),
                FOREIGN KEY(created_by_run_id) REFERENCES runs(id)
            )
            """
        )
        self._set_schema_version(connection, 4)

    def _migrate_v4_to_v5(self, connection: sqlite3.Connection) -> None:
        aesthetic_job_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(aesthetic_jobs)").fetchall()
        }
        if "seed_system_prompt" not in aesthetic_job_columns:
            connection.execute(
                "ALTER TABLE aesthetic_jobs ADD COLUMN seed_system_prompt TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "UPDATE aesthetic_jobs SET seed_system_prompt = seed_refinement_prompt WHERE seed_system_prompt = ''"
            )
        if "compiled_system_prompt" not in aesthetic_job_columns:
            connection.execute("ALTER TABLE aesthetic_jobs ADD COLUMN compiled_system_prompt TEXT")
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET compiled_system_prompt = compiled_gepa_prompt
                WHERE compiled_system_prompt IS NULL AND compiled_gepa_prompt IS NOT NULL
                """
            )

        rollout_columns = {row[1] for row in connection.execute("PRAGMA table_info(rollouts)").fetchall()}
        if "candidate_image_uri" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN candidate_image_uri TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                "UPDATE rollouts SET candidate_image_uri = refined_image_uri WHERE candidate_image_uri = ''"
            )
        if "system_prompt" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''")
            connection.execute(
                "UPDATE rollouts SET system_prompt = refinement_prompt WHERE system_prompt = ''"
            )
        if "generation_mode" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN generation_mode TEXT NOT NULL DEFAULT 'image_conditioned'"
            )

        self._set_schema_version(connection, 5)

    def create_run(self, run_type: RunType, display_name: str, config: dict[str, Any]) -> str:
        if run_type not in VALID_RUN_TYPES:
            raise ValueError(f"Unsupported run type: {run_type}")

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        run_dir = run_artifact_dir(self.artifact_root, run_type, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        created_at = _utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, run_type, display_name, status, config_json, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_type,
                    display_name,
                    "queued",
                    json.dumps(config),
                    str(run_dir),
                    created_at,
                ),
            )
            self._create_typed_run_record(connection, run_id, run_type, config, created_at, str(run_dir))
            connection.commit()
        self.append_run_event(run_id, "INFO", "Run queued for dispatch.")

        return run_id

    def _create_typed_run_record(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        run_type: RunType,
        config: dict[str, Any],
        created_at: str,
        artifact_dir: str,
    ) -> None:
        if run_type == "generation":
            seed_count = config.get("seed_count")
            if seed_count is None and isinstance(config.get("seeds"), list):
                seed_count = len(config["seeds"])
            connection.execute(
                """
                INSERT INTO generation_runs(run_id, prompt_set_id, model_name, seed_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config.get("prompt_set_id") or config.get("prompt_set"),
                    config.get("model_name"),
                    seed_count,
                    created_at,
                ),
            )
            return

        if run_type == "reward_model":
            connection.execute(
                """
                INSERT INTO reward_model_versions(run_id, version_name, base_model, status, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(config.get("version_name") or f"reward-model-{run_id}"),
                    config.get("base_model"),
                    "queued",
                    artifact_dir,
                    created_at,
                ),
            )
            return

        if run_type == "gepa":
            connection.execute(
                """
                INSERT INTO gepa_runs(run_id, parent_reward_model_run_id, status, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config.get("parent_reward_model_run_id"),
                    "queued",
                    artifact_dir,
                    created_at,
                ),
            )
            return

        if run_type == "evaluation":
            connection.execute(
                """
                INSERT INTO evaluation_runs(run_id, baseline_run_id, candidate_run_id, status, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    config.get("baseline_run_id"),
                    config.get("candidate_run_id"),
                    "queued",
                    artifact_dir,
                    created_at,
                ),
            )

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        if status not in VALID_RUN_STATUSES:
            raise ValueError(f"Invalid run status: {status}")
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} does not exist")

        current_status = run["status"]
        valid_transitions: dict[str, set[str]] = {
            "queued": {"running", "cancelled"},
            "running": {"completed", "failed", "cancelled"},
            "completed": set(),
            "failed": set(),
            "cancelled": set(),
        }
        if status != current_status and status not in valid_transitions[current_status]:
            raise ValueError(f"Invalid transition: {current_status} -> {status}")

        updates = {"status": status}
        if status == "running" and run["started_at"] is None:
            updates["started_at"] = _utc_now()
        if status in {"completed", "failed", "cancelled"} and run["finished_at"] is None:
            updates["finished_at"] = _utc_now()

        columns = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [run_id]
        with self._connect() as connection:
            connection.execute(f"UPDATE runs SET {columns} WHERE id = ?", values)
            run_type = run["run_type"]
            if run_type == "reward_model":
                connection.execute(
                    "UPDATE reward_model_versions SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )
            elif run_type == "gepa":
                connection.execute("UPDATE gepa_runs SET status = ? WHERE run_id = ?", (status, run_id))
            elif run_type == "evaluation":
                connection.execute(
                    "UPDATE evaluation_runs SET status = ? WHERE run_id = ?",
                    (status, run_id),
                )
            connection.commit()

    def append_run_event(self, run_id: str, level: str, message: str) -> None:
        if self.get_run(run_id) is None:
            raise ValueError(f"Run {run_id} does not exist")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events (run_id, level, message, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, level, message, _utc_now()),
            )
            connection.commit()

    def list_run_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        if self.get_run(run_id) is None:
            raise ValueError(f"Run {run_id} does not exist")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, level, message, created_at
                FROM run_events
                WHERE run_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, run_type, display_name, status, created_at, started_at, finished_at, artifact_dir
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, run_type, display_name, status, created_at, started_at, finished_at, artifact_dir
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_rating_session(self, name: str) -> str:
        session_id = f"session_{uuid.uuid4().hex[:10]}"
        session_dir = rating_session_artifact_dir(self.artifact_root, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rating_sessions (id, name, status, artifact_dir, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, name, "active", str(session_dir), _utc_now()),
            )
            connection.commit()
        return session_id

    def add_comparison(
        self,
        *,
        session_id: str,
        prompt_text: str,
        left_image_uri: str,
        right_image_uri: str,
        winner: str | None,
        critique: str,
        outcome: RatingOutcome,
    ) -> str:
        session = self.get_rating_session(session_id)
        if session is None:
            raise ValueError(f"Rating session {session_id} does not exist")
        if session["status"] != "active":
            raise ValueError(f"Cannot add comparison to inactive session {session_id}")
        if outcome not in VALID_RATING_OUTCOMES:
            raise ValueError(f"Invalid outcome: {outcome}")
        if outcome == "winner" and winner not in {"left", "right"}:
            raise ValueError("Outcome `winner` requires winner to be `left` or `right`")
        if not critique.strip():
            raise ValueError("Comparison critique cannot be empty")

        comparison_id = f"cmp_{uuid.uuid4().hex[:10]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO comparisons (
                    id, rating_session_id, prompt_text, left_image_uri, right_image_uri,
                    winner, critique, outcome, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison_id,
                    session_id,
                    prompt_text,
                    left_image_uri,
                    right_image_uri,
                    winner,
                    critique,
                    outcome,
                    _utc_now(),
                ),
            )
            connection.commit()
        return comparison_id

    def create_aesthetic_job(self, name: str, description: str, seed_system_prompt: str) -> str:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO aesthetic_jobs (
                    id, name, description, status, seed_system_prompt,
                    active_candidate_id, compiled_system_prompt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    name.strip(),
                    description.strip(),
                    "active",
                    seed_system_prompt.strip(),
                    None,
                    None,
                    created_at,
                    created_at,
                ),
            )
            connection.commit()
        return job_id

    def list_aesthetic_jobs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = (
            "SELECT * FROM aesthetic_jobs ORDER BY created_at DESC"
            if include_archived
            else "SELECT * FROM aesthetic_jobs WHERE status = 'active' ORDER BY created_at DESC"
        )
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def get_aesthetic_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM aesthetic_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_aesthetic_job_policy(
        self, job_id: str, active_candidate_id: str | None, compiled_system_prompt: str | None
    ) -> None:
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET active_candidate_id = ?, compiled_system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (active_candidate_id, compiled_system_prompt, _utc_now(), job_id),
            )
            connection.commit()

    def create_rollout(
        self,
        *,
        job_id: str,
        prompt_text: str,
        intent_text: str,
        baseline_image_uri: str,
        candidate_image_uri: str,
        candidate_id: str | None,
        system_prompt: str,
        generation_mode: str,
        model_config: dict[str, Any],
    ) -> str:
        if generation_mode not in VALID_ROLLOUT_GENERATION_MODES:
            raise ValueError(f"Unsupported rollout generation_mode: {generation_mode}")
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        rollout_id = f"rollout_{uuid.uuid4().hex[:10]}"
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rollouts (
                    id, job_id, comparison_id, prompt_text, intent_text,
                    baseline_image_uri, candidate_image_uri, candidate_id, system_prompt,
                    generation_mode, model_config_json, status, created_at, feedback_completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rollout_id,
                    job_id,
                    None,
                    prompt_text,
                    intent_text,
                    baseline_image_uri,
                    candidate_image_uri,
                    candidate_id,
                    system_prompt,
                    generation_mode,
                    json.dumps(model_config),
                    "generated",
                    created_at,
                    None,
                ),
            )
            connection.commit()
        return rollout_id

    def create_gepa_candidate(
        self,
        *,
        job_id: str,
        parent_candidate_ids: list[str],
        candidate_text: str,
        compiled_prompt: str,
        objective_scores: dict[str, float],
        created_by_run_id: str | None,
    ) -> str:
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        if created_by_run_id is not None and self.get_run(created_by_run_id) is None:
            raise ValueError(f"Run {created_by_run_id} does not exist")

        candidate_id = f"candidate_{uuid.uuid4().hex[:10]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO gepa_candidates(
                    id, job_id, parent_candidate_ids_json, candidate_text,
                    compiled_prompt, objective_scores_json, frontier_member,
                    created_by_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    job_id,
                    json.dumps(parent_candidate_ids),
                    candidate_text,
                    compiled_prompt,
                    json.dumps(objective_scores),
                    0,
                    created_by_run_id,
                    _utc_now(),
                ),
            )
            connection.commit()
        return candidate_id

    def list_gepa_candidates_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, job_id, parent_candidate_ids_json, candidate_text, compiled_prompt,
                       objective_scores_json, frontier_member, created_by_run_id, created_at
                FROM gepa_candidates
                WHERE job_id = ?
                ORDER BY created_at DESC
                """,
                (job_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["parent_candidate_ids"] = json.loads(item.pop("parent_candidate_ids_json"))
            item["objective_scores"] = json.loads(item.pop("objective_scores_json"))
            item["frontier_member"] = bool(item["frontier_member"])
            results.append(item)
        return results

    def set_candidate_frontier_membership(self, candidate_id: str, frontier_member: bool) -> None:
        with self._connect() as connection:
            candidate = connection.execute(
                "SELECT id FROM gepa_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise ValueError(f"GEPA candidate {candidate_id} does not exist")
            connection.execute(
                "UPDATE gepa_candidates SET frontier_member = ? WHERE id = ?",
                (1 if frontier_member else 0, candidate_id),
            )
            connection.commit()

    def promote_job_candidate(self, job_id: str, candidate_id: str) -> None:
        with self._connect() as connection:
            job = connection.execute(
                "SELECT id FROM aesthetic_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise ValueError(f"Aesthetic job {job_id} does not exist")
            candidate = connection.execute(
                "SELECT id, job_id, compiled_prompt FROM gepa_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise ValueError(f"GEPA candidate {candidate_id} does not exist")
            if candidate["job_id"] != job_id:
                raise ValueError("Cannot promote candidate for a different aesthetic job")
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET active_candidate_id = ?, compiled_system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (candidate_id, candidate["compiled_prompt"], _utc_now(), job_id),
            )
            connection.commit()

    def mark_rollout_feedback_complete(self, rollout_id: str, comparison_id: str) -> None:
        completed_at = _utc_now()
        with self._connect() as connection:
            rollout = connection.execute(
                "SELECT id FROM rollouts WHERE id = ?",
                (rollout_id,),
            ).fetchone()
            if rollout is None:
                raise ValueError(f"Rollout {rollout_id} does not exist")
            comparison = connection.execute(
                "SELECT id FROM comparisons WHERE id = ?",
                (comparison_id,),
            ).fetchone()
            if comparison is None:
                raise ValueError(f"Comparison {comparison_id} does not exist")
            connection.execute(
                """
                UPDATE rollouts
                SET status = ?, comparison_id = ?, feedback_completed_at = ?
                WHERE id = ?
                """,
                ("feedback_complete", comparison_id, completed_at, rollout_id),
            )
            connection.commit()

    def list_completed_rollouts_for_job(
        self, job_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT * FROM rollouts WHERE job_id = ? AND status = 'feedback_complete' "
            "ORDER BY feedback_completed_at DESC"
        )
        params: tuple[Any, ...] = (job_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (job_id, limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count_completed_rollouts_for_job(self, job_id: str) -> int:
        with self._connect() as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM rollouts
                WHERE job_id = ? AND status = 'feedback_complete'
                """,
                (job_id,),
            ).fetchone()[0]
        return int(count)

    def list_completed_rollout_ids_for_job(self, job_id: str, limit: int) -> list[str]:
        bounded_limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM rollouts
                WHERE job_id = ? AND status = 'feedback_complete'
                ORDER BY feedback_completed_at DESC
                LIMIT ?
                """,
                (job_id, bounded_limit),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_completed_rollouts_with_feedback(
        self, job_id: str, rollout_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not rollout_ids:
            return []

        placeholders = ", ".join("?" for _ in rollout_ids)
        params: tuple[Any, ...] = (job_id, *rollout_ids)
        query = f"""
            SELECT
                r.id,
                r.job_id,
                r.prompt_text,
                r.intent_text,
                r.baseline_image_uri,
                r.candidate_image_uri,
                r.candidate_id,
                r.system_prompt,
                r.generation_mode,
                r.model_config_json,
                r.status,
                r.created_at,
                r.feedback_completed_at,
                r.comparison_id,
                c.winner,
                c.outcome,
                c.critique
            FROM rollouts r
            JOIN comparisons c ON c.id = r.comparison_id
            WHERE
                r.job_id = ?
                AND r.status = 'feedback_complete'
                AND r.id IN ({placeholders})
            ORDER BY r.feedback_completed_at DESC
        """
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["model_config"] = json.loads(item.pop("model_config_json"))
            items.append(item)
        return items

    def list_recent_comparisons(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, rating_session_id, prompt_text, winner, critique, outcome, created_at
                FROM comparisons
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_rating_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, status, artifact_dir, created_at
                FROM rating_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def schema_version(self) -> int:
        with self._connect() as connection:
            return self._schema_version(connection)

    def integrity_report(self, check_artifacts: bool = True) -> dict[str, list[str]]:
        issues: dict[str, list[str]] = {
            "invalid_runs": [],
            "invalid_status_transitions": [],
            "invalid_rating_sessions": [],
            "invalid_comparisons": [],
            "invalid_rollouts": [],
            "dangling_artifacts": [],
        }
        with self._connect() as connection:
            runs = connection.execute(
                "SELECT id, run_type, status, started_at, finished_at, artifact_dir FROM runs"
            ).fetchall()
            for run in runs:
                run_id = run["id"]
                if run["run_type"] not in VALID_RUN_TYPES:
                    issues["invalid_runs"].append(f"{run_id}: invalid run_type `{run['run_type']}`")
                if run["status"] not in VALID_RUN_STATUSES:
                    issues["invalid_runs"].append(f"{run_id}: invalid status `{run['status']}`")
                if run["finished_at"] is not None and run["started_at"] is None:
                    issues["invalid_status_transitions"].append(
                        f"{run_id}: finished_at set before started_at"
                    )
                if run["status"] in TERMINAL_RUN_STATUSES and run["finished_at"] is None:
                    issues["invalid_status_transitions"].append(
                        f"{run_id}: terminal status without finished_at"
                    )
                if run["status"] == "running" and run["started_at"] is None:
                    issues["invalid_status_transitions"].append(
                        f"{run_id}: running status without started_at"
                    )
                if check_artifacts and not Path(run["artifact_dir"]).exists():
                    issues["dangling_artifacts"].append(
                        f"{run_id}: missing artifact directory `{run['artifact_dir']}`"
                    )

            sessions = connection.execute(
                "SELECT id, status, artifact_dir FROM rating_sessions"
            ).fetchall()
            for session in sessions:
                session_id = session["id"]
                if session["status"] not in VALID_RATING_SESSION_STATUSES:
                    issues["invalid_rating_sessions"].append(
                        f"{session_id}: invalid status `{session['status']}`"
                    )
                if check_artifacts and session["artifact_dir"] and not Path(session["artifact_dir"]).exists():
                    issues["dangling_artifacts"].append(
                        f"{session_id}: missing artifact directory `{session['artifact_dir']}`"
                    )

            comparisons = connection.execute(
                """
                SELECT c.id, c.rating_session_id, c.winner, c.outcome, rs.id AS session_exists
                FROM comparisons c
                LEFT JOIN rating_sessions rs ON rs.id = c.rating_session_id
                """
            ).fetchall()
            for comparison in comparisons:
                comp_id = comparison["id"]
                if comparison["session_exists"] is None:
                    issues["invalid_comparisons"].append(
                        f"{comp_id}: missing rating_session `{comparison['rating_session_id']}`"
                    )
                if comparison["outcome"] not in VALID_RATING_OUTCOMES:
                    issues["invalid_comparisons"].append(
                        f"{comp_id}: invalid outcome `{comparison['outcome']}`"
                    )
                if comparison["outcome"] == "winner" and comparison["winner"] not in {"left", "right"}:
                    issues["invalid_comparisons"].append(
                        f"{comp_id}: winner outcome requires left/right winner"
                    )
            rollouts = connection.execute(
                """
                SELECT r.id, r.job_id, r.status, r.generation_mode, j.id AS job_exists
                FROM rollouts r
                LEFT JOIN aesthetic_jobs j ON j.id = r.job_id
                """
            ).fetchall()
            for rollout in rollouts:
                rollout_id = rollout["id"]
                if rollout["job_exists"] is None:
                    issues["invalid_rollouts"].append(
                        f"{rollout_id}: missing aesthetic_job `{rollout['job_id']}`"
                    )
                if rollout["status"] not in VALID_ROLLOUT_STATUSES:
                    issues["invalid_rollouts"].append(
                        f"{rollout_id}: invalid rollout status `{rollout['status']}`"
                    )
                mode = rollout["generation_mode"] if "generation_mode" in rollout.keys() else None
                if mode is not None and mode not in VALID_ROLLOUT_GENERATION_MODES:
                    issues["invalid_rollouts"].append(
                        f"{rollout_id}: invalid rollout generation_mode `{mode}`"
                    )

        return issues

    def overview_metrics(self) -> dict[str, int]:
        with self._connect() as connection:
            queued = connection.execute("SELECT COUNT(*) FROM runs WHERE status = 'queued'").fetchone()[0]
            running = connection.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0]
            failed = connection.execute("SELECT COUNT(*) FROM runs WHERE status = 'failed'").fetchone()[0]
            pending_reviews = connection.execute(
                "SELECT COUNT(*) FROM rating_sessions WHERE status = 'active'"
            ).fetchone()[0]
            total_comparisons = connection.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]

        return {
            "queued_runs": queued,
            "running_runs": running,
            "failed_runs": failed,
            "active_rating_sessions": pending_reviews,
            "total_comparisons": total_comparisons,
        }

