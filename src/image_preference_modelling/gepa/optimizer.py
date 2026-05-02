from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from image_preference_modelling.gepa.mutation_engine import generate_prompt_mutation
from image_preference_modelling.gepa.scoring import score_rollout_feedback
from image_preference_modelling.storage.state_store import StateStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _select_parent_candidate(
    *,
    job: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str | None, str]:
    requested_parent_id = str(config.get("parent_candidate_id") or "").strip()
    if requested_parent_id:
        requested = next((candidate for candidate in candidates if candidate["id"] == requested_parent_id), None)
        if requested is not None:
            return requested["id"], str(requested["compiled_prompt"])

    evaluated_candidates = [candidate for candidate in candidates if candidate.get("status") == "evaluated"]
    frontier = [candidate for candidate in evaluated_candidates if candidate["frontier_member"]]
    pool = frontier or evaluated_candidates
    if pool:
        seed = config.get("candidate_selection_seed")
        rng = random.Random(seed) if seed is not None else random.Random()
        selected = rng.choice(pool)
        return selected["id"], str(selected["compiled_prompt"])

    parent_candidate_id = config.get("active_candidate_id") or job.get("active_candidate_id")
    base_prompt = (job.get("latest_system_prompt") or job.get("compiled_system_prompt") or "").strip()
    return str(parent_candidate_id) if parent_candidate_id else None, base_prompt

def run_gepa_optimization(
    *,
    run_id: str,
    artifact_dir: Path,
    state_store: StateStore,
    config: dict[str, Any],
    append_event: Callable[[str, str], None],
    is_cancel_requested: Callable[[], bool],
) -> None:
    job_id = str(config.get("job_id", "")).strip()
    minibatch_size = int(config.get("minibatch_size", 0))
    selected_rollout_ids = list(config.get("selected_rollout_ids") or [])
    if not job_id:
        raise ValueError("GEPA run config missing `job_id`")
    if minibatch_size < 1:
        raise ValueError("GEPA run config `minibatch_size` must be >= 1")
    if len(selected_rollout_ids) < minibatch_size:
        raise ValueError("GEPA run config has fewer selected rollouts than minibatch size")

    append_event("INFO", f"Loading job `{job_id}` and selected rollouts.")
    job = state_store.get_aesthetic_job(job_id)
    if job is None:
        raise ValueError(f"Aesthetic job {job_id} not found")

    rollouts = state_store.get_completed_rollouts_with_feedback(job_id, selected_rollout_ids)
    if len(rollouts) < minibatch_size:
        raise ValueError("Not enough completed rollouts found for selected rollout ids")
    rollouts = rollouts[:minibatch_size]

    if is_cancel_requested():
        raise RuntimeError("Cancellation requested before scoring")

    append_event("INFO", f"Scoring {len(rollouts)} completed rollouts.")
    rollout_scores: dict[str, dict[str, float]] = {}
    objective_totals: dict[str, float] = {
        "preference_win": 0.0,
        "feedback_quality": 0.0,
        "intent_preservation": 0.0,
        "composition_preservation": 0.0,
    }
    critiques: list[str] = []
    for rollout in rollouts:
        scores = score_rollout_feedback(rollout)
        rollout_scores[str(rollout["id"])] = scores
        for key, value in scores.items():
            objective_totals[key] = objective_totals.get(key, 0.0) + float(value)
        critiques.append(str(rollout.get("critique", "")))

    n = float(len(rollouts))
    objective_means = {key: value / n for key, value in objective_totals.items()}
    existing_candidates = state_store.list_gepa_candidates_for_job(job_id)
    parent_candidate_id, base_prompt = _select_parent_candidate(
        job=job,
        candidates=existing_candidates,
        config=config,
    )
    requested_backend = str(config.get("optimizer_backend") or "").strip()
    mutation = generate_prompt_mutation(
        parent_prompt=base_prompt or "Refine the image while preserving original intent.",
        job_description=str(job.get("description") or ""),
        critiques=critiques,
        lineage_summary={
            "parent_candidate_id": parent_candidate_id,
            "objective_means": objective_means,
            "existing_candidate_count": len(existing_candidates),
        },
        allow_env_settings=not (
            requested_backend in {"heuristic", "heuristic_fallback"}
            or config.get("openrouter_api_key") == ""
            or config.get("prompt_model") == ""
        ),
    )
    compiled_prompt = mutation.compiled_prompt
    optimizer_backend = mutation.backend
    append_event("INFO", f"Prompt mutation generated with backend `{optimizer_backend}`.")

    parent_candidate_ids = [parent_candidate_id] if parent_candidate_id else []
    candidate_id = state_store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=parent_candidate_ids,
        candidate_text=compiled_prompt,
        compiled_prompt=compiled_prompt,
        objective_scores=objective_means,
        created_by_run_id=run_id,
    )
    frontier_snapshot = state_store.recompute_gepa_frontier_for_job(job_id)
    append_event("INFO", f"GEPA candidate {candidate_id} created as proposed; promotion requires evaluation.")

    checkpoint = {
        "run_id": run_id,
        "job_id": job_id,
        "minibatch_size": minibatch_size,
        "selected_rollout_ids": [str(r["id"]) for r in rollouts],
        "parent_candidate_id": parent_candidate_id,
        "new_candidate_id": candidate_id,
        "new_candidate_status": "proposed",
        "promoted_candidate": False,
        "compiled_prompt": compiled_prompt,
        "optimizer_backend": optimizer_backend,
        "mutation_metadata": mutation.metadata,
        "objective_scores": objective_means,
        "frontier_snapshot": frontier_snapshot,
        "created_at": _utc_now(),
        "rollout_scores": rollout_scores,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    append_event("INFO", f"GEPA checkpoint written to `{artifact_dir / 'checkpoint.json'}`.")
