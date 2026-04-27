from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from image_preference_modelling.gepa.optimizer import run_gepa_optimization
from image_preference_modelling.storage.state_store import StateStore


class JobLauncher:
    """Background dispatcher for run lifecycle orchestration."""

    def __init__(self, state_store: StateStore, max_workers: int = 2) -> None:
        self.state_store = state_store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run-job")
        self._lock = threading.Lock()
        self._active_jobs: dict[str, Future[None]] = {}
        self._cancel_requests: set[str] = set()

    def dispatch_run(self, run_id: str) -> str:
        run = self.state_store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} does not exist")
        if run["status"] != "queued":
            raise ValueError(f"Run {run_id} is not queued (current status: {run['status']})")

        with self._lock:
            if run_id in self._active_jobs:
                raise ValueError(f"Run {run_id} is already running")
            future = self._executor.submit(self._execute_run, run_id)
            self._active_jobs[run_id] = future

        self.state_store.append_run_event(run_id, "INFO", "Run dispatched to worker.")
        return "Run dispatched. Refresh to track progress."

    def cancel_run(self, run_id: str) -> str:
        run = self.state_store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} does not exist")
        if run["status"] in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Run {run_id} is already terminal ({run['status']})")

        with self._lock:
            if run["status"] == "queued":
                self.state_store.update_run_status(run_id, "cancelled")
                self.state_store.append_run_event(run_id, "WARN", "Run cancelled before dispatch.")
                self._write_job_log(run_id)
                return "Run cancelled before dispatch."

            self._cancel_requests.add(run_id)
            self.state_store.append_run_event(run_id, "WARN", "Cancellation requested.")
            return "Cancellation requested."

    def start_run(self, run_id: str) -> None:
        """Backward-compatible alias for dispatch semantics."""
        self.dispatch_run(run_id)

    def _execute_run(self, run_id: str) -> None:
        try:
            run = self.state_store.get_run(run_id)
            if run is None:
                return

            self.state_store.update_run_status(run_id, "running")
            self.state_store.append_run_event(run_id, "INFO", "Worker started.")

            config = json.loads((Path(run["artifact_dir"]) / "config.json").read_text(encoding="utf-8"))
            if config.get("force_fail"):
                raise RuntimeError("forced failure requested by run config")

            if run["run_type"] == "gepa":
                run_gepa_optimization(
                    run_id=run_id,
                    artifact_dir=Path(run["artifact_dir"]),
                    state_store=self.state_store,
                    config=config,
                    append_event=lambda level, message: self.state_store.append_run_event(
                        run_id, level, message
                    ),
                    is_cancel_requested=lambda: self._is_cancel_requested(run_id),
                )
                self.state_store.update_run_status(run_id, "completed")
                self.state_store.append_run_event(run_id, "INFO", "Worker completed successfully.")
                return

            steps = max(1, int(config.get("simulated_steps", 3)))
            sleep_seconds = float(config.get("simulated_step_seconds", 0.02))
            for step in range(1, steps + 1):
                if self._is_cancel_requested(run_id):
                    self.state_store.update_run_status(run_id, "cancelled")
                    self.state_store.append_run_event(
                        run_id, "WARN", f"Worker observed cancellation at step {step}."
                    )
                    return
                self.state_store.append_run_event(run_id, "INFO", f"Processing step {step}/{steps}")
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            self.state_store.update_run_status(run_id, "completed")
            self.state_store.append_run_event(run_id, "INFO", "Worker completed successfully.")
        except Exception as exc:  # noqa: BLE001 - operational failure path
            self.state_store.update_run_status(run_id, "failed")
            self.state_store.append_run_event(run_id, "ERROR", f"Run failed: {exc}")
        finally:
            self._write_job_log(run_id)
            with self._lock:
                self._active_jobs.pop(run_id, None)
                self._cancel_requests.discard(run_id)

    def _is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancel_requests

    def _write_job_log(self, run_id: str) -> None:
        run = self.state_store.get_run(run_id)
        if run is None:
            return
        artifact_dir = Path(run["artifact_dir"])
        events = self.state_store.list_run_events(run_id)
        lines = [
            f"RUN_ID={run_id}",
            f"RUN_TYPE={run['run_type']}",
            f"STATUS={run['status'].upper()}",
            "",
        ]
        lines.extend(f"{event['created_at']} [{event['level']}] {event['message']}" for event in events)
        (artifact_dir / "job.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

