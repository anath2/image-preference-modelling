from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


RunType = Literal["generation", "reward_model", "gepa", "evaluation"]
RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
RatingOutcome = Literal["winner", "both_good", "both_bad", "cant_decide"]
RatingSessionStatus = Literal["active", "archived"]

RUN_TYPE_ARTIFACT_DIR: dict[RunType, str] = {
    "generation": "generation_runs",
    "reward_model": "reward_model_versions",
    "gepa": "gepa_runs",
    "evaluation": "evaluation_runs",
}


@dataclass(frozen=True)
class RunRecord:
    id: str
    run_type: RunType
    display_name: str
    status: RunStatus
    artifact_dir: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class PromptSetRecord:
    id: str
    name: str
    status: str
    artifact_dir: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRunRecord:
    run_id: str
    prompt_set_id: str | None
    model_name: str | None
    seed_count: int | None
    created_at: str

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class RatingSessionRecord:
    id: str
    name: str
    status: RatingSessionStatus
    artifact_dir: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RewardModelVersionRecord:
    run_id: str
    version_name: str
    base_model: str | None
    status: RunStatus
    artifact_dir: str
    created_at: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class GEPARunRecord:
    run_id: str
    parent_reward_model_run_id: str | None
    status: RunStatus
    artifact_dir: str
    created_at: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRunRecord:
    run_id: str
    baseline_run_id: str | None
    candidate_run_id: str | None
    status: RunStatus
    artifact_dir: str
    created_at: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def run_artifact_dir(root: Path, run_type: RunType, run_id: str) -> Path:
    """Canonical artifact directory for run records."""
    return root / RUN_TYPE_ARTIFACT_DIR[run_type] / run_id


def rating_session_artifact_dir(root: Path, session_id: str) -> Path:
    """Canonical artifact directory for rating sessions."""
    return root / "rating_sessions" / session_id


def prompt_set_artifact_dir(root: Path, prompt_set_id: str) -> Path:
    """Canonical artifact directory for prompt sets."""
    return root / "prompt_sets" / prompt_set_id


def artifact_path(base_dir: Path, *parts: str) -> Path:
    """Canonical location helper for files under an artifact directory."""
    return base_dir / Path(*parts)

