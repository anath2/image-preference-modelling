from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from image_preference_modelling.jobs.job_launcher import JobLauncher
from image_preference_modelling.storage.state_store import StateStore


@dataclass(frozen=True)
class AppContext:
    state_store: StateStore
    job_launcher: JobLauncher


def default_context(base_dir: Path | None = None) -> AppContext:
    root = base_dir or Path(".local")
    state_store = StateStore(
        db_path=root / "state" / "cockpit.db",
        artifact_root=root / "artifacts",
    )
    return AppContext(state_store=state_store, job_launcher=JobLauncher(state_store))

