from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_preference_modelling.gepa.reward import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ELO,
    DEFAULT_SCORE,
    blended_candidate_score,
    confidence_from_evidence,
    pairwise_elo_update,
)
from image_preference_modelling.storage.contracts import (
    RatingOutcome,
    RunStatus,
    RunType,
    rating_session_artifact_dir,
    run_artifact_dir,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_VERSION = 10
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
VALID_RUN_STATUSES = {"queued", "running", *TERMINAL_RUN_STATUSES}
VALID_RUN_TYPES = {"generation", "reward_model", "gepa", "evaluation"}
VALID_RATING_SESSION_STATUSES = {"active", "archived"}
VALID_RATING_OUTCOMES = {"winner", "no_clear_winner", "both_good", "both_bad", "cant_decide"}
VALID_AESTHETIC_JOB_STATUSES = {"active", "archived"}
VALID_ROLLOUT_STATUSES = {"generated", "feedback_complete"}
VALID_ROLLOUT_TYPES = {"baseline_candidate", "candidate_comparison", "latest_prompt_check"}
VALID_ROLLOUT_GENERATION_MODES = {"image_conditioned", "text_only"}
VALID_GEPA_CANDIDATE_STATUSES = {"proposed", "evaluating", "evaluated", "archived"}
DEFAULT_GEPA_ENABLE_THRESHOLD = 2


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

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}

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
                baseline_system_prompt TEXT NOT NULL,
                latest_system_prompt TEXT NOT NULL,
                sampling_profile_json TEXT NOT NULL DEFAULT '{}',
                gepa_enable_threshold INTEGER NOT NULL DEFAULT 2,
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
                baseline_system_prompt_snapshot TEXT NOT NULL DEFAULT '',
                latest_system_prompt_snapshot TEXT NOT NULL DEFAULT '',
                prompt_category TEXT,
                selection_mode TEXT,
                llm_score REAL,
                llm_reason TEXT,
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
                elo REAL NOT NULL DEFAULT 1000.0,
                score REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.0,
                judge_metadata_json TEXT NOT NULL DEFAULT '{}',
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

        if current_version == 5:
            self._migrate_v5_to_v6(connection)
            current_version = 6

        if current_version == 6:
            self._migrate_v6_to_v7(connection)
            current_version = 7

        if current_version == 7:
            self._migrate_v7_to_v8(connection)
            current_version = 8

        if current_version == 8:
            self._migrate_v8_to_v9(connection)
            current_version = 9

        if current_version == 9:
            self._migrate_v9_to_v10(connection)
            current_version = 10

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

    def _migrate_v5_to_v6(self, connection: sqlite3.Connection) -> None:
        aesthetic_job_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(aesthetic_jobs)").fetchall()
        }
        if "baseline_system_prompt" not in aesthetic_job_columns:
            connection.execute(
                "ALTER TABLE aesthetic_jobs ADD COLUMN baseline_system_prompt TEXT NOT NULL DEFAULT ''"
            )
        if "latest_system_prompt" not in aesthetic_job_columns:
            connection.execute(
                "ALTER TABLE aesthetic_jobs ADD COLUMN latest_system_prompt TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET latest_system_prompt = seed_system_prompt
                WHERE latest_system_prompt = ''
                """
            )
        if "sampling_profile_json" not in aesthetic_job_columns:
            connection.execute(
                "ALTER TABLE aesthetic_jobs ADD COLUMN sampling_profile_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "gepa_enable_threshold" not in aesthetic_job_columns:
            connection.execute(
                f"ALTER TABLE aesthetic_jobs ADD COLUMN gepa_enable_threshold INTEGER NOT NULL DEFAULT {DEFAULT_GEPA_ENABLE_THRESHOLD}"
            )

        rollout_columns = {row[1] for row in connection.execute("PRAGMA table_info(rollouts)").fetchall()}
        if "baseline_system_prompt_snapshot" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN baseline_system_prompt_snapshot TEXT NOT NULL DEFAULT ''"
            )
        if "latest_system_prompt_snapshot" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN latest_system_prompt_snapshot TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                """
                UPDATE rollouts
                SET latest_system_prompt_snapshot = system_prompt
                WHERE latest_system_prompt_snapshot = ''
                """
            )
        if "prompt_category" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN prompt_category TEXT")

        self._set_schema_version(connection, 6)

    def _migrate_v6_to_v7(self, connection: sqlite3.Connection) -> None:
        rollout_columns = {row[1] for row in connection.execute("PRAGMA table_info(rollouts)").fetchall()}
        if "selection_mode" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN selection_mode TEXT")
        if "llm_score" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN llm_score REAL")
        if "llm_reason" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN llm_reason TEXT")
        self._set_schema_version(connection, 7)

    def _migrate_v7_to_v8(self, connection: sqlite3.Connection) -> None:
        rollout_columns = {row[1] for row in connection.execute("PRAGMA table_info(rollouts)").fetchall()}
        if "rollout_type" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN rollout_type TEXT NOT NULL DEFAULT 'baseline_candidate'"
            )
        if "left_candidate_id" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN left_candidate_id TEXT")
        if "right_candidate_id" not in rollout_columns:
            connection.execute("ALTER TABLE rollouts ADD COLUMN right_candidate_id TEXT")
            connection.execute(
                """
                UPDATE rollouts
                SET right_candidate_id = candidate_id
                WHERE right_candidate_id IS NULL AND candidate_id IS NOT NULL
                """
            )
        if "left_system_prompt_snapshot" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN left_system_prompt_snapshot TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                """
                UPDATE rollouts
                SET left_system_prompt_snapshot = baseline_system_prompt_snapshot
                WHERE left_system_prompt_snapshot = ''
                """
            )
        if "right_system_prompt_snapshot" not in rollout_columns:
            connection.execute(
                "ALTER TABLE rollouts ADD COLUMN right_system_prompt_snapshot TEXT NOT NULL DEFAULT ''"
            )
            connection.execute(
                """
                UPDATE rollouts
                SET right_system_prompt_snapshot = system_prompt
                WHERE right_system_prompt_snapshot = ''
                """
            )
        self._set_schema_version(connection, 8)

    def _migrate_v8_to_v9(self, connection: sqlite3.Connection) -> None:
        candidate_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(gepa_candidates)").fetchall()
        }
        if "status" not in candidate_columns:
            connection.execute(
                "ALTER TABLE gepa_candidates ADD COLUMN status TEXT NOT NULL DEFAULT 'evaluated'"
            )
        if "evaluation_count" not in candidate_columns:
            connection.execute(
                "ALTER TABLE gepa_candidates ADD COLUMN evaluation_count INTEGER NOT NULL DEFAULT 0"
            )
        if "win_count" not in candidate_columns:
            connection.execute("ALTER TABLE gepa_candidates ADD COLUMN win_count INTEGER NOT NULL DEFAULT 0")
        if "loss_count" not in candidate_columns:
            connection.execute("ALTER TABLE gepa_candidates ADD COLUMN loss_count INTEGER NOT NULL DEFAULT 0")
        if "tie_count" not in candidate_columns:
            connection.execute("ALTER TABLE gepa_candidates ADD COLUMN tie_count INTEGER NOT NULL DEFAULT 0")
        self._set_schema_version(connection, 9)

    def _migrate_v9_to_v10(self, connection: sqlite3.Connection) -> None:
        candidate_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(gepa_candidates)").fetchall()
        }
        if "elo" not in candidate_columns:
            connection.execute(
                f"ALTER TABLE gepa_candidates ADD COLUMN elo REAL NOT NULL DEFAULT {DEFAULT_ELO}"
            )
        if "score" not in candidate_columns:
            connection.execute(
                f"ALTER TABLE gepa_candidates ADD COLUMN score REAL NOT NULL DEFAULT {DEFAULT_SCORE}"
            )
        if "confidence" not in candidate_columns:
            connection.execute(
                f"ALTER TABLE gepa_candidates ADD COLUMN confidence REAL NOT NULL DEFAULT {DEFAULT_CONFIDENCE}"
            )
        if "judge_metadata_json" not in candidate_columns:
            connection.execute(
                "ALTER TABLE gepa_candidates ADD COLUMN judge_metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        self._set_schema_version(connection, 10)

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

    def create_aesthetic_job(
        self,
        name: str,
        description: str,
        seed_system_prompt: str,
        *,
        sampling_profile: dict[str, Any] | None = None,
        gepa_enable_threshold: int = DEFAULT_GEPA_ENABLE_THRESHOLD,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        created_at = _utc_now()
        cleaned_seed = seed_system_prompt.strip()
        with self._connect() as connection:
            table_columns = self._table_columns(connection, "aesthetic_jobs")
            payload: dict[str, Any] = {
                "id": job_id,
                "name": name.strip(),
                "description": description.strip(),
                "status": "active",
                "seed_system_prompt": cleaned_seed,
                "baseline_system_prompt": "",
                "latest_system_prompt": cleaned_seed,
                "sampling_profile_json": json.dumps(sampling_profile or {}),
                "gepa_enable_threshold": max(1, int(gepa_enable_threshold)),
                "active_candidate_id": None,
                "compiled_system_prompt": cleaned_seed,
                "created_at": created_at,
                "updated_at": created_at,
            }
            # Backward-compatible population for legacy columns still present in migrated DBs.
            if "seed_refinement_prompt" in table_columns:
                payload["seed_refinement_prompt"] = cleaned_seed
            if "compiled_gepa_prompt" in table_columns:
                payload["compiled_gepa_prompt"] = cleaned_seed
            insert_columns = [col for col in payload.keys() if col in table_columns]
            placeholders = ", ".join("?" for _ in insert_columns)
            column_list = ", ".join(insert_columns)
            connection.execute(
                f"INSERT INTO aesthetic_jobs ({column_list}) VALUES ({placeholders})",
                tuple(payload[col] for col in insert_columns),
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
        return [self._hydrate_aesthetic_job(dict(row)) for row in rows]

    def get_aesthetic_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM aesthetic_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._hydrate_aesthetic_job(dict(row)) if row else None

    def _hydrate_aesthetic_job(self, row: dict[str, Any]) -> dict[str, Any]:
        profile = row.get("sampling_profile_json")
        if isinstance(profile, str):
            try:
                row["sampling_profile"] = json.loads(profile)
            except json.JSONDecodeError:
                row["sampling_profile"] = {}
        return row

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

    def update_aesthetic_job(
        self,
        job_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        sampling_profile: dict[str, Any] | None = None,
        gepa_enable_threshold: int | None = None,
    ) -> None:
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [_utc_now()]
        if name is not None:
            assignments.append("name = ?")
            values.append(name.strip())
        if description is not None:
            assignments.append("description = ?")
            values.append(description.strip())
        if sampling_profile is not None:
            assignments.append("sampling_profile_json = ?")
            values.append(json.dumps(sampling_profile))
        if gepa_enable_threshold is not None:
            assignments.append("gepa_enable_threshold = ?")
            values.append(max(1, int(gepa_enable_threshold)))
        values.append(job_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE aesthetic_jobs SET {', '.join(assignments)} WHERE id = ?",
                tuple(values),
            )
            connection.commit()

    def archive_aesthetic_job(self, job_id: str) -> None:
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        with self._connect() as connection:
            connection.execute(
                "UPDATE aesthetic_jobs SET status = 'archived', updated_at = ? WHERE id = ?",
                (_utc_now(), job_id),
            )
            connection.commit()

    def rollover_job_system_prompt(
        self,
        job_id: str,
        *,
        latest_system_prompt: str,
        active_candidate_id: str | None = None,
    ) -> None:
        job = self.get_aesthetic_job(job_id)
        if job is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        previous_latest = str(job.get("latest_system_prompt") or "")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET baseline_system_prompt = ?, latest_system_prompt = ?, compiled_system_prompt = ?,
                    active_candidate_id = COALESCE(?, active_candidate_id), updated_at = ?
                WHERE id = ?
                """,
                (
                    previous_latest,
                    latest_system_prompt.strip(),
                    latest_system_prompt.strip(),
                    active_candidate_id,
                    _utc_now(),
                    job_id,
                ),
            )
            connection.commit()

    def create_rollout(
        self,
        *,
        rollout_id: str | None = None,
        job_id: str,
        prompt_text: str,
        intent_text: str,
        baseline_image_uri: str,
        candidate_image_uri: str,
        candidate_id: str | None,
        system_prompt: str,
        baseline_system_prompt_snapshot: str = "",
        latest_system_prompt_snapshot: str = "",
        rollout_type: str = "baseline_candidate",
        left_candidate_id: str | None = None,
        right_candidate_id: str | None = None,
        left_system_prompt_snapshot: str | None = None,
        right_system_prompt_snapshot: str | None = None,
        prompt_category: str | None = None,
        selection_mode: str | None = None,
        llm_score: float | None = None,
        llm_reason: str | None = None,
        generation_mode: str,
        model_config: dict[str, Any],
    ) -> str:
        if rollout_type not in VALID_ROLLOUT_TYPES:
            raise ValueError(f"Unsupported rollout_type: {rollout_type}")
        if generation_mode not in VALID_ROLLOUT_GENERATION_MODES:
            raise ValueError(f"Unsupported rollout generation_mode: {generation_mode}")
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        resolved_rollout_id = rollout_id or f"rollout_{uuid.uuid4().hex[:10]}"
        created_at = _utc_now()
        with self._connect() as connection:
            table_columns = self._table_columns(connection, "rollouts")
            payload: dict[str, Any] = {
                "id": resolved_rollout_id,
                "job_id": job_id,
                "comparison_id": None,
                "prompt_text": prompt_text,
                "intent_text": intent_text,
                "baseline_image_uri": baseline_image_uri,
                "candidate_image_uri": candidate_image_uri,
                "candidate_id": candidate_id,
                "system_prompt": system_prompt,
                "baseline_system_prompt_snapshot": baseline_system_prompt_snapshot,
                "latest_system_prompt_snapshot": latest_system_prompt_snapshot,
                "rollout_type": rollout_type,
                "left_candidate_id": left_candidate_id,
                "right_candidate_id": right_candidate_id if right_candidate_id is not None else candidate_id,
                "left_system_prompt_snapshot": (
                    left_system_prompt_snapshot
                    if left_system_prompt_snapshot is not None
                    else baseline_system_prompt_snapshot
                ),
                "right_system_prompt_snapshot": (
                    right_system_prompt_snapshot if right_system_prompt_snapshot is not None else system_prompt
                ),
                "prompt_category": prompt_category,
                "selection_mode": selection_mode,
                "llm_score": llm_score,
                "llm_reason": llm_reason,
                "generation_mode": generation_mode,
                "model_config_json": json.dumps(model_config),
                "status": "generated",
                "created_at": created_at,
                "feedback_completed_at": None,
            }
            # Backward-compatible population for legacy columns still present in migrated DBs.
            if "refined_image_uri" in table_columns:
                payload["refined_image_uri"] = candidate_image_uri
            if "refinement_prompt" in table_columns:
                payload["refinement_prompt"] = system_prompt
            insert_columns = [col for col in payload.keys() if col in table_columns]
            placeholders = ", ".join("?" for _ in insert_columns)
            column_list = ", ".join(insert_columns)
            connection.execute(
                f"INSERT INTO rollouts ({column_list}) VALUES ({placeholders})",
                tuple(payload[col] for col in insert_columns),
            )
            connection.commit()
        return resolved_rollout_id

    def create_gepa_candidate(
        self,
        *,
        job_id: str,
        parent_candidate_ids: list[str],
        candidate_text: str,
        compiled_prompt: str,
        objective_scores: dict[str, float],
        created_by_run_id: str | None,
        status: str = "proposed",
    ) -> str:
        if status not in VALID_GEPA_CANDIDATE_STATUSES:
            raise ValueError(f"Unsupported GEPA candidate status: {status}")
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
                    elo, score, confidence, judge_metadata_json,
                    status, evaluation_count, win_count, loss_count, tie_count,
                    created_by_run_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    job_id,
                    json.dumps(parent_candidate_ids),
                    candidate_text,
                    compiled_prompt,
                    json.dumps(objective_scores),
                    0,
                    DEFAULT_ELO,
                    DEFAULT_SCORE,
                    DEFAULT_CONFIDENCE,
                    json.dumps({}),
                    status,
                    0,
                    0,
                    0,
                    0,
                    created_by_run_id,
                    _utc_now(),
                ),
            )
            connection.commit()
        return candidate_id

    def list_gepa_candidates_for_job(
        self, job_id: str, statuses: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if statuses is not None:
            invalid_statuses = [status for status in statuses if status not in VALID_GEPA_CANDIDATE_STATUSES]
            if invalid_statuses:
                raise ValueError(f"Unsupported GEPA candidate status: {invalid_statuses[0]}")
        with self._connect() as connection:
            where_clause = "WHERE job_id = ?"
            params: tuple[Any, ...] = (job_id,)
            if statuses:
                placeholders = ", ".join("?" for _ in statuses)
                where_clause += f" AND status IN ({placeholders})"
                params = (job_id, *statuses)
            rows = connection.execute(
                f"""
                SELECT id, job_id, parent_candidate_ids_json, candidate_text, compiled_prompt,
                       objective_scores_json, frontier_member, status, evaluation_count,
                       win_count, loss_count, tie_count, elo, score, confidence,
                       judge_metadata_json, created_by_run_id, created_at
                FROM gepa_candidates
                {where_clause}
                ORDER BY created_at DESC
                """,
                params,
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["parent_candidate_ids"] = json.loads(item.pop("parent_candidate_ids_json"))
            item["objective_scores"] = json.loads(item.pop("objective_scores_json"))
            item["judge_metadata"] = json.loads(item.pop("judge_metadata_json") or "{}")
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

    def update_gepa_candidate_status(self, candidate_id: str, status: str) -> None:
        if status not in VALID_GEPA_CANDIDATE_STATUSES:
            raise ValueError(f"Unsupported GEPA candidate status: {status}")
        with self._connect() as connection:
            candidate = connection.execute(
                "SELECT id FROM gepa_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise ValueError(f"GEPA candidate {candidate_id} does not exist")
            connection.execute(
                "UPDATE gepa_candidates SET status = ? WHERE id = ?",
                (status, candidate_id),
            )
            connection.commit()

    def update_candidate_feedback_stats(
        self,
        *,
        winner_candidate_id: str | None,
        loser_candidate_id: str | None,
        tied_candidate_ids: list[str] | None = None,
        winner_margin: float = 1.0,
        critique_confidence: float = 1.0,
        judge_metadata: dict[str, Any] | None = None,
    ) -> None:
        tied_ids = [candidate_id for candidate_id in (tied_candidate_ids or []) if candidate_id]
        with self._connect() as connection:
            candidate_ids = [candidate_id for candidate_id in [winner_candidate_id, loser_candidate_id, *tied_ids] if candidate_id]
            if candidate_ids:
                placeholders = ", ".join("?" for _ in candidate_ids)
                existing = {
                    str(row["id"])
                    for row in connection.execute(
                        f"SELECT id FROM gepa_candidates WHERE id IN ({placeholders})",
                        tuple(candidate_ids),
                    ).fetchall()
                }
                missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in existing]
                if missing:
                    raise ValueError(f"GEPA candidate {missing[0]} does not exist")
            if winner_candidate_id and loser_candidate_id:
                rows = {
                    str(row["id"]): row
                    for row in connection.execute(
                        "SELECT id, elo FROM gepa_candidates WHERE id IN (?, ?)",
                        (winner_candidate_id, loser_candidate_id),
                    ).fetchall()
                }
                elo_update = pairwise_elo_update(
                    left_elo=float(rows[winner_candidate_id]["elo"]),
                    right_elo=float(rows[loser_candidate_id]["elo"]),
                    winner="left",
                    winner_margin=winner_margin,
                )
                connection.execute(
                    "UPDATE gepa_candidates SET elo = ? WHERE id = ?",
                    (elo_update.left_elo, winner_candidate_id),
                )
                connection.execute(
                    "UPDATE gepa_candidates SET elo = ? WHERE id = ?",
                    (elo_update.right_elo, loser_candidate_id),
                )
            if winner_candidate_id:
                connection.execute(
                    """
                    UPDATE gepa_candidates
                    SET evaluation_count = evaluation_count + 1,
                        win_count = win_count + 1,
                        status = 'evaluated'
                    WHERE id = ?
                    """,
                    (winner_candidate_id,),
                )
            if loser_candidate_id:
                connection.execute(
                    """
                    UPDATE gepa_candidates
                    SET evaluation_count = evaluation_count + 1,
                        loss_count = loss_count + 1,
                        status = 'evaluated'
                    WHERE id = ?
                    """,
                    (loser_candidate_id,),
                )
            for candidate_id in tied_ids:
                connection.execute(
                    """
                    UPDATE gepa_candidates
                    SET evaluation_count = evaluation_count + 1,
                        tie_count = tie_count + 1,
                        status = 'evaluated'
                    WHERE id = ?
                    """,
                    (candidate_id,),
                )
            affected_ids = sorted(set(candidate_ids))
            for candidate_id in affected_ids:
                row = connection.execute(
                    """
                    SELECT evaluation_count, win_count, tie_count, elo, judge_metadata_json
                    FROM gepa_candidates
                    WHERE id = ?
                    """,
                    (candidate_id,),
                ).fetchone()
                if row is None:
                    continue
                evaluation_count = int(row["evaluation_count"] or 0)
                win_count = int(row["win_count"] or 0)
                tie_count = int(row["tie_count"] or 0)
                elo = float(row["elo"] or DEFAULT_ELO)
                judge_summary = self._updated_judge_summary(
                    json.loads(row["judge_metadata_json"] or "{}"),
                    winner_margin=winner_margin if candidate_id in {winner_candidate_id, loser_candidate_id} else 0.0,
                    critique_confidence=critique_confidence,
                    judge_metadata=judge_metadata,
                )
                judge_count = int(judge_summary.get("judge_count") or 0)
                avg_confidence = (
                    float(judge_summary.get("critique_confidence_total") or 0.0) / float(judge_count)
                    if judge_count
                    else 0.0
                )
                avg_margin = (
                    float(judge_summary.get("winner_margin_total") or 0.0) / float(judge_count)
                    if judge_count
                    else 0.0
                )
                confidence = confidence_from_evidence(
                    evaluation_count=evaluation_count,
                    average_critique_confidence=avg_confidence,
                )
                blended_score = blended_candidate_score(
                    elo=elo,
                    confidence=confidence,
                    average_margin_quality=avg_margin,
                )
                win_rate = (
                    (float(win_count) + 0.5 * float(tie_count)) / float(evaluation_count)
                    if evaluation_count
                    else 0.0
                )
                scores = {
                    "candidate_win_rate": win_rate,
                    "evaluation_coverage": min(1.0, float(evaluation_count) / 5.0),
                    "human_preference": win_rate,
                    "evaluation_confidence": confidence,
                    "critique_margin": avg_margin,
                    "blended_score": blended_score,
                }
                connection.execute(
                    """
                    UPDATE gepa_candidates
                    SET objective_scores_json = ?, score = ?, confidence = ?, judge_metadata_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(scores), blended_score, confidence, json.dumps(judge_summary), candidate_id),
                )
            connection.commit()

    def _updated_judge_summary(
        self,
        existing: dict[str, Any],
        *,
        winner_margin: float,
        critique_confidence: float,
        judge_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        summary = dict(existing)
        summary["judge_count"] = int(summary.get("judge_count") or 0) + 1
        summary["winner_margin_total"] = float(summary.get("winner_margin_total") or 0.0) + max(
            0.0, min(1.0, float(winner_margin))
        )
        summary["critique_confidence_total"] = float(
            summary.get("critique_confidence_total") or 0.0
        ) + max(0.0, min(1.0, float(critique_confidence)))
        if judge_metadata:
            recent = list(summary.get("recent_judgements") or [])
            recent.append(judge_metadata)
            summary["recent_judgements"] = recent[-5:]
        return summary

    def recompute_gepa_frontier_for_job(self, job_id: str) -> list[dict[str, Any]]:
        candidates = self.list_gepa_candidates_for_job(job_id, statuses=["evaluated"])

        def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
            keys = set(a) | set(b)
            return (
                all(float(a.get(key, 0.0)) >= float(b.get(key, 0.0)) for key in keys)
                and any(float(a.get(key, 0.0)) > float(b.get(key, 0.0)) for key in keys)
            )

        frontier_ids: set[str] = set()
        for candidate in candidates:
            candidate_scores = candidate["objective_scores"]
            is_dominated = any(
                other["id"] != candidate["id"]
                and dominates(other["objective_scores"], candidate_scores)
                for other in candidates
            )
            if not is_dominated:
                frontier_ids.add(str(candidate["id"]))

        with self._connect() as connection:
            connection.execute(
                "UPDATE gepa_candidates SET frontier_member = 0 WHERE job_id = ?",
                (job_id,),
            )
            for candidate in candidates:
                connection.execute(
                    "UPDATE gepa_candidates SET frontier_member = ? WHERE id = ?",
                    (1 if candidate["id"] in frontier_ids else 0, candidate["id"]),
                )
            connection.commit()

        return [
            {"candidate_id": candidate["id"], "frontier_member": candidate["id"] in frontier_ids}
            for candidate in candidates
        ]

    def promote_job_candidate(self, job_id: str, candidate_id: str) -> None:
        with self._connect() as connection:
            job = connection.execute(
                "SELECT id FROM aesthetic_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise ValueError(f"Aesthetic job {job_id} does not exist")
            candidate = connection.execute(
                "SELECT id, job_id, compiled_prompt, status, frontier_member FROM gepa_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise ValueError(f"GEPA candidate {candidate_id} does not exist")
            if candidate["job_id"] != job_id:
                raise ValueError("Cannot promote candidate for a different aesthetic job")
            if candidate["status"] != "evaluated":
                raise ValueError("Cannot promote a GEPA candidate before it is evaluated")
            if not bool(candidate["frontier_member"]):
                raise ValueError("Cannot promote a GEPA candidate that is not on the frontier")
            job_row = connection.execute(
                "SELECT latest_system_prompt FROM aesthetic_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            previous_latest = str(job_row["latest_system_prompt"] if job_row else "")
            connection.execute(
                """
                UPDATE aesthetic_jobs
                SET active_candidate_id = ?, compiled_system_prompt = ?, baseline_system_prompt = ?,
                    latest_system_prompt = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    candidate_id,
                    candidate["compiled_prompt"],
                    previous_latest,
                    candidate["compiled_prompt"],
                    _utc_now(),
                    job_id,
                ),
            )
            connection.commit()

    def promote_best_frontier_candidate(self, job_id: str) -> str:
        self.recompute_gepa_frontier_for_job(job_id)
        candidates = [
            candidate
            for candidate in self.list_gepa_candidates_for_job(job_id, statuses=["evaluated"])
            if candidate["frontier_member"]
        ]
        if not candidates:
            raise ValueError(f"Aesthetic job {job_id} has no evaluated frontier candidates")

        def ranking_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
            scores = candidate["objective_scores"]
            return (
                float(candidate.get("score") or scores.get("blended_score", 0.0)),
                int(candidate.get("evaluation_count") or 0),
                str(candidate["created_at"]),
            )

        selected = max(candidates, key=ranking_key)
        self.promote_job_candidate(job_id, str(selected["id"]))
        return str(selected["id"])

    def get_best_training_candidate(
        self,
        job_id: str,
        *,
        min_evaluations: int = 3,
        min_confidence: float = 0.5,
    ) -> dict[str, Any] | None:
        candidates = self.list_gepa_candidates_for_job(job_id, statuses=["evaluated"])
        eligible = [
            candidate
            for candidate in candidates
            if int(candidate.get("evaluation_count") or 0) >= min_evaluations
            and float(candidate.get("confidence") or 0.0) >= min_confidence
        ]
        if not eligible:
            return None

        def ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
            return (
                float(candidate.get("score") or 0.0),
                float(candidate.get("elo") or DEFAULT_ELO),
                float(candidate.get("confidence") or 0.0),
                str(candidate["created_at"]),
            )

        return max(eligible, key=ranking_key)

    def count_pending_gepa_candidates_for_job(self, job_id: str) -> int:
        with self._connect() as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM gepa_candidates
                WHERE job_id = ? AND status IN ('proposed', 'evaluating')
                """,
                (job_id,),
            ).fetchone()[0]
        return int(count or 0)

    def archive_pending_gepa_candidates_for_job(self, job_id: str) -> int:
        if self.get_aesthetic_job(job_id) is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE gepa_candidates
                SET status = 'archived'
                WHERE job_id = ? AND status IN ('proposed', 'evaluating')
                """,
                (job_id,),
            )
            connection.commit()
        return int(cursor.rowcount or 0)

    def count_active_gepa_runs_for_job(self, job_id: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT config_json
                FROM runs
                WHERE run_type = 'gepa' AND status IN ('queued', 'running')
                """
            ).fetchall()
        count = 0
        for row in rows:
            try:
                config = json.loads(row["config_json"])
            except json.JSONDecodeError:
                continue
            if str(config.get("job_id") or "").strip() == job_id:
                count += 1
        return count

    def _latest_completed_gepa_finished_at_for_job(
        self, connection: sqlite3.Connection, job_id: str
    ) -> str | None:
        rows = connection.execute(
            """
            SELECT config_json, finished_at
            FROM runs
            WHERE run_type = 'gepa'
                AND status = 'completed'
                AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            """
        ).fetchall()
        for row in rows:
            try:
                config = json.loads(row["config_json"])
            except json.JSONDecodeError:
                continue
            if str(config.get("job_id") or "").strip() == job_id:
                return str(row["finished_at"])
        return None

    def get_gepa_gate_status(self, job_id: str) -> dict[str, Any]:
        job = self.get_aesthetic_job(job_id)
        if job is None:
            raise ValueError(f"Aesthetic job {job_id} does not exist")
        threshold = max(1, int(job.get("gepa_enable_threshold") or DEFAULT_GEPA_ENABLE_THRESHOLD))
        with self._connect() as connection:
            total_completed = connection.execute(
                """
                SELECT COUNT(*)
                FROM rollouts
                WHERE job_id = ? AND status = 'feedback_complete'
                """,
                (job_id,),
            ).fetchone()[0]
            last_completed_at = self._latest_completed_gepa_finished_at_for_job(connection, job_id)
            if last_completed_at is None:
                new_feedback = total_completed
            else:
                new_feedback = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM rollouts
                    WHERE job_id = ?
                        AND status = 'feedback_complete'
                        AND feedback_completed_at > ?
                    """,
                    (job_id, last_completed_at),
                ).fetchone()[0]
        new_feedback_count = int(new_feedback or 0)
        return {
            "completed_feedback_count": int(total_completed or 0),
            "new_feedback_count": new_feedback_count,
            "threshold": threshold,
            "enabled": new_feedback_count >= threshold,
            "last_gepa_completed_at": last_completed_at,
        }

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

    def list_gepa_eligible_rollout_ids_for_job(self, job_id: str, limit: int) -> list[str]:
        bounded_limit = max(1, int(limit))
        with self._connect() as connection:
            last_completed_at = self._latest_completed_gepa_finished_at_for_job(connection, job_id)
            if last_completed_at is None:
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
            else:
                rows = connection.execute(
                    """
                    SELECT id
                    FROM rollouts
                    WHERE job_id = ?
                        AND status = 'feedback_complete'
                        AND feedback_completed_at > ?
                    ORDER BY feedback_completed_at DESC
                    LIMIT ?
                    """,
                    (job_id, last_completed_at, bounded_limit),
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
                r.rollout_type,
                r.left_candidate_id,
                r.right_candidate_id,
                r.left_system_prompt_snapshot,
                r.right_system_prompt_snapshot,
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

    def list_rollouts_for_job(self, job_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.*,
                    c.winner,
                    c.outcome,
                    c.critique
                FROM rollouts r
                LEFT JOIN comparisons c ON c.id = r.comparison_id
                WHERE r.job_id = ?
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (job_id, max(1, int(limit))),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["model_config"] = json.loads(item.pop("model_config_json"))
            results.append(item)
        return results

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
                SELECT r.id, r.job_id, r.status, r.rollout_type, r.generation_mode, j.id AS job_exists
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
                rollout_type = rollout["rollout_type"] if "rollout_type" in rollout.keys() else None
                if rollout_type is not None and rollout_type not in VALID_ROLLOUT_TYPES:
                    issues["invalid_rollouts"].append(
                        f"{rollout_id}: invalid rollout_type `{rollout_type}`"
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

