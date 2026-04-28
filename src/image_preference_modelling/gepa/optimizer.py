from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from image_preference_modelling.gepa.scoring import score_rollout_feedback
from image_preference_modelling.storage.state_store import StateStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_compiled_prompt(
    current_prompt: str | None,
    critiques: list[str],
) -> str:
    base_prompt = (current_prompt or "").strip() or "Refine the image while preserving original intent."
    base = base_prompt.split("\n\nFeedback reflections:")[0].strip()
    condensed = "; ".join(c.strip() for c in critiques if c.strip())
    if not condensed:
        return base
    return f"{base}\n\nFeedback reflections: {condensed}"


def _optimize_with_dspy_gepa(
    *,
    base_prompt: str,
    critiques: list[str],
    minibatch_size: int,
    config: dict[str, Any],
) -> str:
    import dspy

    model_name = (
        str(config.get("dspy_model") or "").strip()
        or os.getenv("DSPY_MODEL", "").strip()
        or os.getenv("PROMPT_MODEL", "").strip()
    )
    api_key = str(config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY", "")).strip()
    api_base = (
        str(config.get("openrouter_base_url") or os.getenv("OPENROUTER_BASE_URL", "")).strip()
        or "https://openrouter.ai/api/v1"
    )
    if not model_name or not api_key:
        raise RuntimeError("Missing DSPy/OpenRouter configuration for GEPA optimization")

    lm = dspy.LM(
        model=model_name,
        api_key=api_key,
        api_base=api_base.rstrip("/"),
        temperature=0,
        max_tokens=2048,
    )
    dspy.configure(lm=lm)

    class PolicyRewrite(dspy.Signature):
        """Improve a system prompt policy based on user feedback."""

        current_policy = dspy.InputField()
        feedback = dspy.InputField()
        improved_policy = dspy.OutputField()

    class PolicyStudent(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proposer = dspy.Predict(PolicyRewrite)

        def forward(self, current_policy: str, feedback: str):  # type: ignore[override]
            return self.proposer(current_policy=current_policy, feedback=feedback)

    trainset = [
        dspy.Example(
            current_policy=base_prompt,
            feedback=critique.strip(),
            improved_policy=base_prompt,
        ).with_inputs("current_policy", "feedback")
        for critique in critiques
        if critique.strip()
    ]
    if not trainset:
        raise RuntimeError("No critique feedback available for DSPy GEPA optimization")

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):  # noqa: ANN001, ARG001
        improved = str(getattr(pred, "improved_policy", "")).strip()
        return 1.0 if improved else 0.0

    teleprompter = dspy.GEPA(
        metric=metric,
        auto="light",
        reflection_minibatch_size=max(1, minibatch_size),
        candidate_selection_strategy="pareto",
        reflection_lm=lm,
    )
    optimized = teleprompter.compile(
        PolicyStudent(),
        trainset=trainset,
        valset=trainset,
    )
    prediction = optimized(
        current_policy=base_prompt,
        feedback="; ".join(c.strip() for c in critiques if c.strip()),
    )
    improved_prompt = str(getattr(prediction, "improved_policy", "")).strip()
    if not improved_prompt:
        raise RuntimeError("DSPy GEPA did not produce an improved policy")
    return improved_prompt


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
    base_prompt = (job.get("compiled_system_prompt") or "").strip()
    optimizer_backend = str(config.get("optimizer_backend", "dspy_gepa")).strip() or "dspy_gepa"
    if optimizer_backend == "dspy_gepa":
        try:
            compiled_prompt = _optimize_with_dspy_gepa(
                base_prompt=base_prompt or "Refine the image while preserving original intent.",
                critiques=critiques,
                minibatch_size=minibatch_size,
                config=config,
            )
            append_event("INFO", "DSPy GEPA optimization completed.")
        except Exception as exc:  # noqa: BLE001
            append_event("WARN", f"DSPy GEPA unavailable; falling back to heuristic policy update ({exc}).")
            compiled_prompt = _build_compiled_prompt(job.get("compiled_system_prompt"), critiques)
            optimizer_backend = "heuristic_fallback"
    else:
        compiled_prompt = _build_compiled_prompt(job.get("compiled_system_prompt"), critiques)

    parent_candidate_id = config.get("active_candidate_id") or job.get("active_candidate_id")
    parent_candidate_ids = [parent_candidate_id] if parent_candidate_id else []
    candidate_id = state_store.create_gepa_candidate(
        job_id=job_id,
        parent_candidate_ids=parent_candidate_ids,
        candidate_text=compiled_prompt,
        compiled_prompt=compiled_prompt,
        objective_scores=objective_means,
        created_by_run_id=run_id,
    )
    state_store.set_candidate_frontier_membership(candidate_id, True)
    state_store.promote_job_candidate(job_id, candidate_id)

    checkpoint = {
        "run_id": run_id,
        "job_id": job_id,
        "minibatch_size": minibatch_size,
        "selected_rollout_ids": [str(r["id"]) for r in rollouts],
        "parent_candidate_id": parent_candidate_id,
        "new_candidate_id": candidate_id,
        "compiled_prompt": compiled_prompt,
        "optimizer_backend": optimizer_backend,
        "objective_scores": objective_means,
        "frontier_snapshot": [{"candidate_id": candidate_id, "frontier_member": True}],
        "created_at": _utc_now(),
        "rollout_scores": rollout_scores,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    append_event("INFO", f"GEPA checkpoint written to `{artifact_dir / 'checkpoint.json'}`.")
