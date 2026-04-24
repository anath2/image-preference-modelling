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


SCHEMA_VERSION = 2
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
VALID_RUN_STATUSES = {"queued", "running", *TERMINAL_RUN_STATUSES}
VALID_RUN_TYPES = {"generation", "reward_model", "gepa", "evaluation"}
VALID_RATING_SESSION_STATUSES = {"active", "archived"}
VALID_RATING_OUTCOMES = {"winner", "both_good", "both_bad", "cant_decide"}


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
            self._create_schema_v2(connection)
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

    def _create_schema_v2(self, connection: sqlite3.Connection) -> None:
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

