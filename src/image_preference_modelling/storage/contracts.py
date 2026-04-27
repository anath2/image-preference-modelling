from __future__ import annotations

from pathlib import Path
from typing import Literal


RunType = Literal["generation", "reward_model", "gepa", "evaluation"]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
RatingOutcome = Literal["winner", "both_good", "both_bad", "cant_decide"]

RUN_TYPE_ARTIFACT_DIR: dict[RunType, str] = {
    "generation": "generation_runs",
    "reward_model": "reward_model_versions",
    "gepa": "gepa_runs",
    "evaluation": "evaluation_runs",
}


def run_artifact_dir(root: Path, run_type: RunType, run_id: str) -> Path:
    """Canonical artifact directory for run records."""
    return root / RUN_TYPE_ARTIFACT_DIR[run_type] / run_id


def rating_session_artifact_dir(root: Path, session_id: str) -> Path:
    """Canonical artifact directory for rating sessions."""
    return root / "rating_sessions" / session_id

