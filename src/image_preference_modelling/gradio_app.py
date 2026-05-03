from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import gradio as gr

from image_preference_modelling.app_context import AppContext, default_context
from image_preference_modelling.config import ImageGenerationModelSettings
from image_preference_modelling.generation_pipeline import (
    DEFAULT_PROMPT_SOURCE_ROOT,
    DEFAULT_PROMPT_CANDIDATE_COUNT,
    DEFAULT_TIMEOUT_SECONDS,
    build_candidate_system_prompt,
    generate_candidate_image_from_openrouter,
    generate_image_from_openrouter,
    rollout_image_dir,
    sample_prompt_for_job,
    save_generated_image,
)
from image_preference_modelling.gepa.critique_judge import judge_critique


AUTO_MUTATION_MIN_TOTAL_FEEDBACK = 3


def _workflow_output_dir() -> Path:
    output_dir = Path(".local") / "artifacts" / "ui_workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _system_prompt_preview(value: str, *, max_chars: int = 80) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "<empty>"
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3]}..."


def _winner_to_storage_outcome(winner: str) -> tuple[str | None, str]:
    if winner in {"baseline", "left"}:
        return "left", "winner"
    if winner in {"candidate", "right"}:
        return "right", "winner"
    if winner in {"no_clear_winner", "both_good", "both_bad", "cant_decide"}:
        return None, "no_clear_winner"
    raise ValueError(f"Unsupported winner value: {winner}")


def _job_choices(jobs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(f"{job['name']} ({job['id']})", job["id"]) for job in jobs]


def _resolve_active_system_prompt(job: dict[str, Any]) -> str:
    return (job.get("latest_system_prompt") or job.get("compiled_system_prompt") or job["seed_system_prompt"]).strip()


def _build_gepa_run_config(
    *,
    job_id: str,
    minibatch_size: int,
    selected_rollout_ids: list[str],
    active_candidate_id: str | None,
    compiled_prompt: str | None,
) -> dict[str, Any]:
    prompt_model = os.getenv("PROMPT_MODEL", "").strip()
    return {
        "job_id": job_id,
        "minibatch_size": minibatch_size,
        "selected_rollout_ids": selected_rollout_ids,
        "active_candidate_id": active_candidate_id,
        "compiled_prompt": compiled_prompt,
        "optimizer_backend": "prompt_mutation",
        "prompt_model": prompt_model or None,
    }


def build_app(context: AppContext | None = None) -> gr.Blocks:
    ctx = context or default_context()
    output_dir = _workflow_output_dir()

    with gr.Blocks(title="Gradio Operator Cockpit") as app:
        gr.Markdown("# Image Preference Workflow")
        gr.Markdown(
            "Sample a prompt, prepare a left/right matchup, generate both images, then score."
        )

        prompt_category_state = gr.State(value="")
        prompt_selection_mode_state = gr.State(value="")
        prompt_llm_score_state = gr.State(value="")
        prompt_llm_reason_state = gr.State(value="")
        baseline_path_state = gr.State(value="")
        candidate_path_state = gr.State(value="")
        train_left_candidate_id_state = gr.State(value="")
        train_right_candidate_id_state = gr.State(value="")
        train_left_system_prompt_state = gr.State(value="")
        train_right_system_prompt_state = gr.State(value="")
        train_rollout_type_state = gr.State(value="baseline_candidate")
        session_id_state = gr.State(value="")
        active_job_id_state = gr.State(value="")
        active_rollout_id_state = gr.State(value="")
        latest_gepa_run_id_state = gr.State(value="")
        latest_check_baseline_path_state = gr.State(value="")
        latest_check_candidate_path_state = gr.State(value="")

        workflow_status = gr.Markdown("Ready. Start with `Sample Prompt`.")

        with gr.Tabs():
            with gr.Tab("Jobs"):
                gr.Markdown("## Aesthetic Job")
                with gr.Row():
                    selected_job = gr.Dropdown(label="Active Job", choices=[], value=None)
                    refresh_jobs_btn = gr.Button("Refresh Jobs")
                    use_selected_job_btn = gr.Button("Use Selected Job")
                create_job_name = gr.Textbox(label="New Job Name")
                create_job_description = gr.Textbox(label="New Job Description")
                create_job_seed_prompt = gr.Textbox(label="Seed System Prompt")
                create_job_category = gr.Dropdown(
                    label="Guided Sampling Category",
                    choices=["portrait", "outdoor_landscape", "cityscape"],
                    value="portrait",
                )
                create_job_threshold = gr.Number(label="GEPA Enable Threshold", value=2, minimum=1, precision=0)
                create_job_btn = gr.Button("Create Job")
                update_job_name = gr.Textbox(label="Update Job Name")
                update_job_description = gr.Textbox(label="Update Job Description")
                update_job_category = gr.Dropdown(
                    label="Update Sampling Category",
                    choices=["portrait", "outdoor_landscape", "cityscape"],
                    value="portrait",
                )
                update_job_threshold = gr.Number(label="Update GEPA Threshold", value=2, minimum=1, precision=0)
                update_job_btn = gr.Button("Update Selected Job")
                archive_job_btn = gr.Button("Archive Selected Job")
                active_job_name = gr.Textbox(label="Selected Job Name", interactive=False)
                compiled_prompt_view = gr.Textbox(label="Latest System Prompt", interactive=False)

            with gr.Tab("Train / Compare"):
                gr.Markdown("## Training Comparison")
                gr.Markdown("Use the selected job to prepare a fixed left/right prompt matchup and submit feedback.")
                with gr.Group(visible=False) as rollout_workflow_group:
                    sampled_prompt = gr.Textbox(
                        label="Prompt",
                        interactive=True,
                        placeholder="Sample a prompt or type your own prompt here.",
                    )
                    baseline_candidate_label = gr.Textbox(label="Left Candidate", interactive=False)
                    candidate_candidate_label = gr.Textbox(label="Right Candidate", interactive=False)
                    baseline_prompt_text = gr.Textbox(label="Left System Prompt", interactive=False)
                    candidate_prompt_text = gr.Textbox(label="Right System Prompt", interactive=False)

                    with gr.Row():
                        sample_prompt_btn = gr.Button("Sample Prompt")
                        prepare_matchup_btn = gr.Button("Prepare Matchup")
                        generate_pair_btn = gr.Button("Generate Left / Right Images")

                    with gr.Row():
                        baseline_image = gr.Image(label="Left", type="filepath")
                        candidate_image = gr.Image(label="Right", type="filepath")

                    winner_choice = gr.Radio(
                        choices=["left", "right", "no_clear_winner"],
                        value="no_clear_winner",
                        label="Winner",
                    )
                    critique_text = gr.Textbox(label="Critique")
                    submit_score_btn = gr.Button("Submit Score")

            with gr.Tab("Prompt Evolution"):
                gr.Markdown("## Prompt Mutation")
                completed_feedback_count = gr.Textbox(label="Completed Feedback Count", interactive=False, value="0")
                minibatch_size = gr.Number(
                    label="Mutation Feedback Batch Size",
                    value=3,
                    minimum=1,
                    maximum=10,
                    precision=0,
                )
                run_gepa_btn = gr.Button("Generate Mutation Now")
                archive_pending_candidates_btn = gr.Button("Archive Pending Candidates")
                promote_frontier_btn = gr.Button("Promote Best Frontier Candidate")
                refresh_gepa_status_btn = gr.Button("Refresh Mutation Status")
                show_gepa_logs_btn = gr.Button("Show Mutation Logs")
                gepa_poll_timer = gr.Timer(value=2.0, active=False)
                latest_gepa_run_status = gr.Textbox(
                    label="Latest Mutation Run Status",
                    interactive=False,
                    value="No mutation run yet.",
                )
                gepa_run_logs = gr.Textbox(
                    label="Mutation Run Logs",
                    interactive=False,
                    lines=14,
                    max_lines=24,
                    value="No mutation logs yet.",
                )

            with gr.Tab("Best Candidate Check"):
                gr.Markdown("## Best Candidate Check")
                gr.Markdown(
                    "Generate a one-off no-system baseline beside the best evaluated training candidate. "
                    "These checks are stored separately from completed training feedback."
                )
                latest_check_prompt = gr.Textbox(
                    label="Best Candidate Check Prompt",
                    interactive=True,
                    placeholder="Type a prompt to compare the best training candidate against baseline.",
                )
                generate_latest_check_btn = gr.Button("Generate Best Candidate Check")
                latest_check_baseline_prompt = gr.Textbox(
                    label="Best Check Baseline System Prompt",
                    interactive=False,
                )
                latest_check_candidate_prompt = gr.Textbox(
                    label="Best Check Candidate System Prompt",
                    interactive=False,
                )
                with gr.Row():
                    latest_check_baseline_image = gr.Image(label="Best Check Baseline", type="filepath")
                    latest_check_candidate_image = gr.Image(label="Best Check Candidate", type="filepath")

            with gr.Tab("Rollout Inspector"):
                gr.Markdown("## Rollout Inspector")
                with gr.Row():
                    inspector_job = gr.Dropdown(label="Inspect Job", choices=[], value=None)
                    refresh_inspector_jobs_btn = gr.Button("Refresh Inspector Jobs")
                refresh_rollouts_btn = gr.Button("Load Rollouts")
                inspector_rollout = gr.Dropdown(label="Rollout", choices=[], value=None)
                rollout_metadata = gr.Textbox(label="Rollout Metadata", interactive=False, lines=14, max_lines=24)
                with gr.Row():
                    inspector_baseline_image = gr.Image(label="Inspector Baseline", type="filepath")
                    inspector_candidate_image = gr.Image(label="Inspector Candidate", type="filepath")

        def _refresh_job_choices() -> tuple[gr.Dropdown, str]:
            jobs = ctx.state_store.list_aesthetic_jobs()
            return gr.update(choices=_job_choices(jobs)), "Job list refreshed."

        def _gepa_button_state(job_id: str) -> tuple[Any, str, str, str, Any]:
            if not job_id.strip():
                return gr.update(interactive=False), "Select and activate an aesthetic job first.", "No mutation run yet.", "", gr.update(active=False)
            gate = ctx.state_store.get_gepa_gate_status(job_id)
            completed_count = ctx.state_store.count_completed_rollouts_for_job(job_id)
            pending_candidates = ctx.state_store.count_pending_gepa_candidates_for_job(job_id)
            active_runs = ctx.state_store.count_active_gepa_runs_for_job(job_id)
            if (
                gate["enabled"]
                and completed_count >= AUTO_MUTATION_MIN_TOTAL_FEEDBACK
                and pending_candidates == 0
                and active_runs == 0
            ):
                job = ctx.state_store.get_aesthetic_job(job_id)
                selected_rollout_ids = ctx.state_store.list_gepa_eligible_rollout_ids_for_job(
                    job_id,
                    limit=min(3, int(gate["new_feedback_count"])),
                )
                if job is not None and selected_rollout_ids:
                    run_config = _build_gepa_run_config(
                        job_id=job_id,
                        minibatch_size=len(selected_rollout_ids),
                        selected_rollout_ids=selected_rollout_ids,
                        active_candidate_id=job.get("active_candidate_id"),
                        compiled_prompt=job.get("latest_system_prompt") or job.get("compiled_system_prompt"),
                    )
                    run_id = ctx.state_store.create_run(
                        run_type="gepa",
                        display_name=f"Prompt Mutation ({job['name']})",
                        config=run_config,
                    )
                    dispatch_message = ctx.job_launcher.dispatch_run(run_id)
                    return (
                        gr.update(interactive=True),
                        f"Mutation auto-started from new feedback: `{run_id}`. {dispatch_message}",
                        f"{run_id} (queued)",
                        run_id,
                        gr.update(active=True),
                    )
            message = (
                "Mutation ready: "
                f"{gate['new_feedback_count']} new feedback items meet threshold {gate['threshold']}."
                if gate["enabled"]
                else (
                    "Mutation waiting: "
                    f"{gate['new_feedback_count']} new feedback items since last mutation; "
                    f"threshold={gate['threshold']}."
                )
            )
            if pending_candidates:
                message = f"Mutation paused: {pending_candidates} proposed/evaluating candidate(s) need feedback."
            elif active_runs:
                message = "Mutation already queued or running."
            elif completed_count < AUTO_MUTATION_MIN_TOTAL_FEEDBACK:
                message = (
                    f"Mutation waiting: need at least {AUTO_MUTATION_MIN_TOTAL_FEEDBACK} total feedback items; "
                    f"currently {completed_count}."
                )
            return gr.update(interactive=bool(gate["enabled"])), message, "No mutation run yet.", "", gr.update(active=False)

        def _create_job(
            name: str,
            description: str,
            seed_prompt: str,
            category: str,
            threshold: int | float,
        ) -> tuple[gr.Dropdown, str]:
            if not name.strip() or not seed_prompt.strip():
                return gr.update(), "Job name and seed system prompt are required."
            ctx.state_store.create_aesthetic_job(
                name,
                description,
                seed_prompt,
                sampling_profile={"category": (category or "").strip()},
                gepa_enable_threshold=int(threshold),
            )
            jobs = ctx.state_store.list_aesthetic_jobs()
            return gr.update(choices=_job_choices(jobs)), "Aesthetic job created."

        def _use_selected_job(job_id: str | None) -> tuple[str, str, str, str, str, str, Any, str]:
            selected = (job_id or "").strip()
            if not selected:
                return "", "", "", "", "", "Select a job first.", gr.update(visible=False), "0"
            job = ctx.state_store.get_aesthetic_job(selected)
            if job is None or job["status"] != "active":
                return "", "", "", "", "", "Selected job is unavailable.", gr.update(visible=False), "0"
            latest_prompt = str(job.get("latest_system_prompt") or job.get("compiled_system_prompt") or "")
            completed_count = ctx.state_store.count_completed_rollouts_for_job(selected)
            return (
                selected,
                job["name"],
                latest_prompt,
                "",
                "",
                f"Using aesthetic job `{job['name']}`.",
                gr.update(visible=True),
                str(completed_count),
            )

        def _update_selected_job(
            active_job_id: str,
            name: str,
            description: str,
            category: str,
            threshold: int | float,
        ) -> str:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "Select and activate an aesthetic job first."
            ctx.state_store.update_aesthetic_job(
                selected_job_id,
                name=name.strip() or None,
                description=description.strip() or None,
                sampling_profile={"category": (category or "").strip()},
                gepa_enable_threshold=int(threshold),
            )
            return "Selected job updated."

        def _archive_selected_job(active_job_id: str) -> tuple[str, str]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "", "Select and activate an aesthetic job first."
            ctx.state_store.archive_aesthetic_job(selected_job_id)
            return "", "Selected job archived."

        def _run_gepa_optimization(
            active_job_id: str,
            minibatch_value: int | float,
        ) -> tuple[str, str, str, str, Any]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "No mutation run yet.", "0", "Select and activate an aesthetic job first.", "", gr.update(active=False)
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return "No mutation run yet.", "0", "Selected job is unavailable.", "", gr.update(active=False)
            chosen_minibatch = int(minibatch_value)
            if chosen_minibatch < 1 or chosen_minibatch > 10:
                return (
                    "No mutation run yet.",
                    str(ctx.state_store.count_completed_rollouts_for_job(selected_job_id)),
                    "Minibatch size must be between 1 and 10.",
                    "",
                    gr.update(active=False),
                )
            completed_count = ctx.state_store.count_completed_rollouts_for_job(selected_job_id)
            gate = ctx.state_store.get_gepa_gate_status(selected_job_id)
            if not gate["enabled"]:
                return (
                    "No mutation run yet.",
                    str(completed_count),
                    (
                        "Mutation is waiting: "
                        f"{gate['new_feedback_count']} new feedback items since last mutation; "
                        f"threshold={gate['threshold']}."
                    ),
                    "",
                    gr.update(active=False),
                )
            selected_rollout_ids = ctx.state_store.list_gepa_eligible_rollout_ids_for_job(
                selected_job_id,
                limit=chosen_minibatch,
            )
            if len(selected_rollout_ids) < chosen_minibatch:
                return (
                    "No mutation run yet.",
                    str(completed_count),
                    (
                        f"Need at least {chosen_minibatch} new completed rollouts since last mutation; "
                        f"currently {len(selected_rollout_ids)}."
                    ),
                    "",
                    gr.update(active=False),
                )
            run_config = _build_gepa_run_config(
                job_id=selected_job_id,
                minibatch_size=chosen_minibatch,
                selected_rollout_ids=selected_rollout_ids,
                active_candidate_id=job.get("active_candidate_id"),
                compiled_prompt=job.get("latest_system_prompt") or job.get("compiled_system_prompt"),
            )
            run_id = ctx.state_store.create_run(
                run_type="gepa",
                display_name=f"Prompt Mutation ({job['name']})",
                config=run_config,
            )
            dispatch_message = ctx.job_launcher.dispatch_run(run_id)
            run = ctx.state_store.get_run(run_id)
            status = run["status"] if run is not None else "queued"
            return (
                f"{run_id} ({status})",
                str(completed_count),
                f"Prompt mutation started. {dispatch_message}",
                run_id,
                gr.update(active=True),
            )

        def _archive_pending_candidates(active_job_id: str) -> tuple[str, str, str, str, Any]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "Select and activate an aesthetic job first.", "No mutation run yet.", "0", "", gr.update(active=False)
            archived_count = ctx.state_store.archive_pending_gepa_candidates_for_job(selected_job_id)
            completed_count = str(ctx.state_store.count_completed_rollouts_for_job(selected_job_id))
            if archived_count == 0:
                message = "No pending candidates to archive."
            else:
                message = f"Archived {archived_count} pending candidate(s). Mutation can resume when the feedback gate is ready."
            return message, "No mutation run yet.", completed_count, "", gr.update(active=False)

        def _refresh_gepa_status(
            run_id: str,
            active_job_id: str,
        ) -> tuple[str, str, str, str, str, str, Any]:
            active_job = active_job_id.strip()
            if not run_id.strip():
                return (
                    "No mutation run yet.",
                    "0",
                    "No mutation run has been started yet.",
                    "",
                    "",
                    "",
                    gr.update(active=False),
                )
            run = ctx.state_store.get_run(run_id.strip())
            if run is None:
                return "No mutation run yet.", "0", "Latest mutation run no longer exists.", "", "", "", gr.update(active=False)
            status = run["status"]
            status_text = f"{run['id']} ({status})"
            completed_count = "0"
            latest_prompt = ""
            if active_job:
                job = ctx.state_store.get_aesthetic_job(active_job)
                if job is not None:
                    completed_count = str(ctx.state_store.count_completed_rollouts_for_job(active_job))
                    latest_prompt = str(
                        job.get("latest_system_prompt") or job.get("compiled_system_prompt") or ""
                    )
            if status == "completed":
                message = "Mutation run completed. Proposed candidate is ready for evaluation."
                timer_update = gr.update(active=False)
            elif status == "failed":
                events = ctx.state_store.list_run_events(run["id"], limit=20)
                error_event = next((e for e in reversed(events) if e["level"] == "ERROR"), None)
                error_text = error_event["message"] if error_event else "No error details available."
                message = f"Mutation run failed: {error_text}"
                timer_update = gr.update(active=False)
            elif status == "cancelled":
                message = "Mutation run was cancelled."
                timer_update = gr.update(active=False)
            else:
                message = f"Mutation run is {status}. Refresh again shortly."
                timer_update = gr.update(active=True)
            return status_text, completed_count, message, latest_prompt, "", "", timer_update

        def _show_gepa_logs(run_id: str) -> tuple[str, str]:
            selected_run = run_id.strip()
            if not selected_run:
                return "No mutation logs yet.", "No mutation run has been started yet."
            run = ctx.state_store.get_run(selected_run)
            if run is None:
                return "No mutation logs yet.", "Latest mutation run no longer exists."
            events = ctx.state_store.list_run_events(selected_run, limit=300)
            if not events:
                return "No mutation logs yet.", "No run events are available yet."
            lines = [
                f"{event['created_at']} [{event['level']}] {event['message']}"
                for event in events
            ]
            return "\n".join(lines), f"Showing logs for `{selected_run}`."

        def _promote_best_frontier_candidate(active_job_id: str) -> tuple[str, str, str, str]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "", "", "", "Select and activate an aesthetic job first."
            try:
                candidate_id = ctx.state_store.promote_best_frontier_candidate(selected_job_id)
            except ValueError as exc:
                return "", "", "", str(exc)
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return "", "", "", "Selected job is unavailable."
            return (
                str(job.get("latest_system_prompt") or ""),
                "",
                "",
                f"Promoted frontier candidate `{candidate_id}`.",
            )

        def _refresh_inspector_jobs() -> tuple[gr.Dropdown, str]:
            jobs = ctx.state_store.list_aesthetic_jobs(include_archived=True)
            return gr.update(choices=_job_choices(jobs)), "Inspector jobs refreshed."

        def _load_rollout_choices(job_id: str | None) -> tuple[gr.Dropdown, str]:
            selected = (job_id or "").strip()
            if not selected:
                return gr.update(choices=[], value=None), "Select a job to inspect rollouts."
            rollouts = ctx.state_store.list_rollouts_for_job(selected, limit=300)
            choices = [
                (
                    f"{item['id']} ({item['status']}) [{item.get('prompt_category') or 'uncategorized'}]",
                    item["id"],
                )
                for item in rollouts
            ]
            if not choices:
                return gr.update(choices=[], value=None), "No rollouts found for selected job."
            return gr.update(choices=choices, value=choices[0][1]), f"Loaded {len(choices)} rollouts."

        def _show_rollout_details(rollout_id: str | None, job_id: str | None) -> tuple[str | None, str | None, str, str]:
            selected_rollout = (rollout_id or "").strip()
            selected_job = (job_id or "").strip()
            if not selected_job:
                return None, None, "", "Select a job first."
            if not selected_rollout:
                return None, None, "", "Select a rollout to inspect."
            rollouts = ctx.state_store.list_rollouts_for_job(selected_job, limit=300)
            item = next((row for row in rollouts if row["id"] == selected_rollout), None)
            if item is None:
                return None, None, "", "Selected rollout no longer exists."
            metadata = {
                "id": item["id"],
                "job_id": item["job_id"],
                "status": item["status"],
                "prompt_text": item["prompt_text"],
                "prompt_category": item.get("prompt_category"),
                "selection_mode": item.get("selection_mode"),
                "llm_score": item.get("llm_score"),
                "llm_reason": item.get("llm_reason"),
                "system_prompt": item.get("system_prompt"),
                "baseline_system_prompt_snapshot": item.get("baseline_system_prompt_snapshot"),
                "latest_system_prompt_snapshot": item.get("latest_system_prompt_snapshot"),
                "winner": item.get("winner"),
                "outcome": item.get("outcome"),
                "critique": item.get("critique"),
                "created_at": item.get("created_at"),
                "feedback_completed_at": item.get("feedback_completed_at"),
            }
            baseline_uri = str(item.get("baseline_image_uri") or "")
            candidate_uri = str(item.get("candidate_image_uri") or "")
            baseline_path = Path(baseline_uri) if baseline_uri else None
            candidate_path = Path(candidate_uri) if candidate_uri else None
            baseline_value = baseline_uri if baseline_path and baseline_path.exists() else None
            candidate_value = candidate_uri if candidate_path and candidate_path.exists() else None
            missing_parts: list[str] = []
            if baseline_uri and baseline_value is None:
                missing_parts.append("baseline image missing on disk")
            if candidate_uri and candidate_value is None:
                missing_parts.append("candidate image missing on disk")
            status_suffix = f" ({'; '.join(missing_parts)})" if missing_parts else ""
            return (
                baseline_value,
                candidate_value,
                json.dumps(metadata, indent=2),
                f"Showing rollout `{selected_rollout}`{status_suffix}.",
            )

        def _select_training_matchup(job: dict[str, Any]) -> dict[str, str | None]:
            job_id = str(job["id"])
            proposed = ctx.state_store.list_gepa_candidates_for_job(job_id, statuses=["proposed"])
            evaluating = ctx.state_store.list_gepa_candidates_for_job(job_id, statuses=["evaluating"])
            evaluated = ctx.state_store.list_gepa_candidates_for_job(job_id, statuses=["evaluated"])
            frontier = [candidate for candidate in evaluated if candidate["frontier_member"]]
            pending = proposed or evaluating
            if pending:
                right = random.choice(pending)
                left_pool = [candidate for candidate in frontier if candidate["id"] != right["id"]]
                if not left_pool:
                    left_pool = [candidate for candidate in evaluated if candidate["id"] != right["id"]]
                left = random.choice(left_pool) if left_pool else None
                return {
                    "rollout_type": "candidate_comparison" if left is not None else "baseline_candidate",
                    "left_candidate_id": None if left is None else str(left["id"]),
                    "right_candidate_id": str(right["id"]),
                    "left_prompt": "" if left is None else str(left["compiled_prompt"]),
                    "right_prompt": str(right["compiled_prompt"]),
                }

            active_candidate_id = str(job.get("active_candidate_id") or "").strip()
            active_candidate = next(
                (candidate for candidate in evaluated if candidate["id"] == active_candidate_id),
                None,
            )
            right_prompt = (
                str(active_candidate["compiled_prompt"])
                if active_candidate is not None
                else _resolve_active_system_prompt(job)
            )
            return {
                "rollout_type": "baseline_candidate",
                "left_candidate_id": None,
                "right_candidate_id": active_candidate_id or None,
                "left_prompt": "",
                "right_prompt": right_prompt,
            }

        def _prepare_training_matchup(
            prompt: str,
            active_job_id: str,
        ) -> tuple[str, str, str, str, str, str, str, str, str, None, None, str, str, str, str]:
            cleaned_prompt = prompt.strip()
            selected_job_id = active_job_id.strip()
            if not cleaned_prompt:
                return "", "", "", "", "", "", "", "", "baseline_candidate", None, None, "", "", "", "Enter or sample a prompt first."
            if not selected_job_id:
                return "", "", "", "", "", "", "", "", "baseline_candidate", None, None, "", "", "", "Select and activate an aesthetic job first."
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return "", "", "", "", "", "", "", "", "baseline_candidate", None, None, "", "", "", "Selected job is unavailable."
            matchup = _select_training_matchup(job)
            left_candidate_id = str(matchup["left_candidate_id"] or "")
            right_candidate_id = str(matchup["right_candidate_id"] or "")
            left_prompt = str(matchup["left_prompt"] or "")
            right_prompt = str(matchup["right_prompt"] or "")
            left_label = left_candidate_id or "No-system baseline"
            right_label = right_candidate_id or "Active/latest prompt"
            left_prompt_display = left_prompt or "<no system prompt>"
            return (
                left_candidate_id,
                right_candidate_id,
                left_prompt,
                right_prompt,
                left_label,
                right_label,
                left_prompt_display,
                right_prompt,
                str(matchup["rollout_type"] or "baseline_candidate"),
                None,
                None,
                "",
                "",
                "",
                "Matchup prepared. Generate the left image, then the right image.",
            )

        def _generate_latest_prompt_check(
            prompt: str,
            active_job_id: str,
        ) -> tuple[str | None, str | None, str, str, str, str, str]:
            cleaned_prompt = prompt.strip()
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return None, None, "", "", "", "", "Select and activate an aesthetic job first."
            if not cleaned_prompt:
                return None, None, "", "", "", "", "Enter a prompt for the best candidate check."
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return None, None, "", "", "", "", "Selected job is unavailable."

            best_candidate = ctx.state_store.get_best_training_candidate(selected_job_id)
            if best_candidate is None:
                return None, None, "", "", "", "", "No evaluated training candidate is available yet."
            best_prompt = str(best_candidate["compiled_prompt"])
            evaluation_count = int(best_candidate.get("evaluation_count") or 0)
            confidence = float(best_candidate.get("confidence") or 0.0)
            evidence_note = (
                f"Evidence is limited ({evaluation_count} eval(s), confidence {confidence:.2f}); "
                "use this as a sanity check, not a promotion signal."
                if evaluation_count < 3 or confidence < 0.5
                else f"Evidence looks stronger ({evaluation_count} eval(s), confidence {confidence:.2f})."
            )

            settings = ImageGenerationModelSettings.from_env()
            rollout_id = f"rollout_{uuid.uuid4().hex[:10]}"
            rollout_dir = rollout_image_dir(selected_job_id, rollout_id)
            rollout_dir.mkdir(parents=True, exist_ok=True)
            baseline_data_url = generate_image_from_openrouter(
                cleaned_prompt,
                settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            baseline_path = save_generated_image(
                baseline_data_url,
                rollout_dir,
                stem="best-check-baseline",
            )
            system_prompt = build_candidate_system_prompt(
                original_prompt=cleaned_prompt,
                regeneration_instructions=best_prompt,
            )
            candidate_data_url = generate_candidate_image_from_openrouter(
                cleaned_prompt,
                system_prompt,
                settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            candidate_path = save_generated_image(
                candidate_data_url,
                rollout_dir,
                stem="best-check-candidate",
            )
            ctx.state_store.create_rollout(
                rollout_id=rollout_id,
                job_id=selected_job_id,
                prompt_text=cleaned_prompt,
                intent_text=cleaned_prompt,
                baseline_image_uri=str(baseline_path),
                candidate_image_uri=str(candidate_path),
                candidate_id=best_candidate["id"],
                system_prompt=system_prompt,
                baseline_system_prompt_snapshot=str(job.get("baseline_system_prompt") or ""),
                latest_system_prompt_snapshot=str(job.get("latest_system_prompt") or ""),
                rollout_type="latest_prompt_check",
                left_candidate_id=None,
                right_candidate_id=best_candidate["id"],
                left_system_prompt_snapshot=str(job.get("baseline_system_prompt") or ""),
                right_system_prompt_snapshot=system_prompt,
                generation_mode="text_only",
                model_config={"image_model": settings.image_model},
            )
            return (
                str(baseline_path),
                str(candidate_path),
                str(baseline_path),
                str(candidate_path),
                "<no system prompt>",
                system_prompt,
                (
                    f"Best candidate check generated as rollout `{rollout_id}` using `{best_candidate['id']}`. "
                    f"{evidence_note}"
                ),
            )

        def _sample_prompt(active_job_id: str) -> tuple[str, str, str, str, str, str]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "", "", "", "", "", "Select and activate an aesthetic job first."
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return "", "", "", "", "", "Selected job is unavailable."
            selected, category, selection_mode, llm_score, llm_reason = sample_prompt_for_job(
                sampling_profile=job.get("sampling_profile"),
                job_description=str(job.get("description") or ""),
                prompt_source_root=DEFAULT_PROMPT_SOURCE_ROOT,
                candidate_count=DEFAULT_PROMPT_CANDIDATE_COUNT,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            message = (
                f"Prompt sampled ({category}). You can edit it, then click `Prepare Matchup`."
                if category
                else "Prompt sampled. You can edit it, then click `Prepare Matchup`."
            )
            return (
                selected,
                category or "",
                selection_mode,
                "" if llm_score is None else str(llm_score),
                llm_reason or "",
                message,
            )

        def _generate_training_pair(
            prompt: str,
            active_job_id: str,
            left_candidate_id: str,
            right_candidate_id: str,
            left_system_prompt_state: str,
            right_system_prompt_state: str,
            rollout_type: str,
            prompt_category: str,
            prompt_selection_mode: str,
            prompt_llm_score: str,
            prompt_llm_reason: str,
        ) -> tuple[str | None, str | None, str, str, str, str, str, str, str]:
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                return None, None, "", "", "", left_system_prompt_state, left_system_prompt_state or "", right_system_prompt_state, "Sample or enter a prompt first."
            if not active_job_id.strip():
                return None, None, "", "", "", left_system_prompt_state, left_system_prompt_state or "", right_system_prompt_state, "Select and activate an aesthetic job first."
            left_prompt = left_system_prompt_state.strip()
            right_prompt = right_system_prompt_state.strip()
            if not right_prompt:
                return None, None, "", "", "", left_system_prompt_state, left_system_prompt_state or "", right_system_prompt_state, "Prepare a matchup first."
            job = ctx.state_store.get_aesthetic_job(active_job_id.strip())
            if job is None:
                return None, None, "", "", "", left_system_prompt_state, left_system_prompt_state or "", right_system_prompt_state, "Selected job is unavailable."

            settings = ImageGenerationModelSettings.from_env()
            rollout_id = f"rollout_{uuid.uuid4().hex[:10]}"
            rollout_dir = rollout_image_dir(active_job_id.strip(), rollout_id)
            rollout_dir.mkdir(parents=True, exist_ok=True)
            left_system_prompt = (
                build_candidate_system_prompt(
                    original_prompt=cleaned_prompt,
                    regeneration_instructions=left_prompt,
                )
                if left_prompt
                else None
            )
            baseline_data_url = generate_image_from_openrouter(
                cleaned_prompt,
                settings,
                system_prompt=left_system_prompt,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            baseline_path = save_generated_image(
                baseline_data_url,
                rollout_dir,
                stem="baseline",
            )
            system_prompt = build_candidate_system_prompt(
                original_prompt=cleaned_prompt,
                regeneration_instructions=right_prompt,
            )
            candidate_data_url = generate_candidate_image_from_openrouter(
                cleaned_prompt,
                system_prompt,
                settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            candidate_path = save_generated_image(
                candidate_data_url,
                rollout_dir,
                stem="candidate",
            )
            ctx.state_store.create_rollout(
                rollout_id=rollout_id,
                job_id=active_job_id.strip(),
                prompt_text=cleaned_prompt,
                intent_text=cleaned_prompt,
                baseline_image_uri=str(baseline_path),
                candidate_image_uri=str(candidate_path),
                candidate_id=right_candidate_id.strip() or None,
                system_prompt=system_prompt,
                baseline_system_prompt_snapshot=str(job.get("baseline_system_prompt") or ""),
                latest_system_prompt_snapshot=str(job.get("latest_system_prompt") or ""),
                rollout_type=rollout_type.strip() or "baseline_candidate",
                left_candidate_id=left_candidate_id.strip() or None,
                right_candidate_id=right_candidate_id.strip() or None,
                left_system_prompt_snapshot=left_system_prompt or "",
                right_system_prompt_snapshot=system_prompt,
                prompt_category=prompt_category.strip() or None,
                selection_mode=prompt_selection_mode.strip() or None,
                llm_score=float(prompt_llm_score) if prompt_llm_score.strip() else None,
                llm_reason=prompt_llm_reason.strip() or None,
                generation_mode="text_only",
                model_config={"image_model": settings.image_model},
            )
            selected_right_candidate_id = right_candidate_id.strip()
            if selected_right_candidate_id:
                right_candidate = next(
                    (
                        candidate
                        for candidate in ctx.state_store.list_gepa_candidates_for_job(active_job_id.strip())
                        if candidate["id"] == selected_right_candidate_id
                    ),
                    None,
                )
                if right_candidate is not None and right_candidate["status"] == "proposed":
                    ctx.state_store.update_gepa_candidate_status(selected_right_candidate_id, "evaluating")
            return (
                str(baseline_path),
                str(candidate_path),
                str(baseline_path),
                str(candidate_path),
                rollout_id,
                left_system_prompt or "",
                left_system_prompt or "<no system prompt>",
                right_prompt,
                (
                    "Left/right images generated. Pick a winner and submit score. "
                    f"Left prompt: `{_system_prompt_preview(left_system_prompt or '')}`; "
                    f"right prompt: `{_system_prompt_preview(system_prompt)}`."
                ),
            )

        def _submit_score(
            prompt: str,
            baseline_path: str,
            candidate_path: str,
            winner: str,
            critique: str,
            session_id: str,
            active_rollout_id: str,
            active_job_id: str,
            left_candidate_id: str,
            right_candidate_id: str,
            left_system_prompt: str,
            right_system_prompt: str,
            rollout_type: str,
            left_candidate_label: str,
            right_candidate_label: str,
        ) -> tuple[Any, ...]:
            cleaned_prompt = prompt.strip()
            left_uri = baseline_path.strip()
            right_uri = candidate_path.strip()
            def preserve(message: str) -> tuple[Any, ...]:
                return (
                    message,
                    session_id,
                    active_rollout_id,
                    "0",
                    left_candidate_id,
                    right_candidate_id,
                    left_system_prompt,
                    right_system_prompt,
                    rollout_type,
                    left_uri or None,
                    right_uri or None,
                    left_uri,
                    right_uri,
                    left_candidate_label,
                    right_candidate_label,
                    critique,
                )

            if not cleaned_prompt:
                return preserve("Prompt is required before scoring.")
            if not left_uri or not right_uri:
                return preserve("Generate both left and right images before scoring.")
            if left_uri == right_uri:
                return preserve("Left and right image URIs must differ.")
            if not active_rollout_id.strip():
                return preserve("Generate a right image to create a rollout before scoring.")
            if not critique.strip():
                return preserve("Critique is required before scoring.")

            active_session_id = session_id.strip()
            if not active_session_id:
                active_session_id = ctx.state_store.create_rating_session("ui-workflow-session")
            winner_value, outcome = _winner_to_storage_outcome(winner)

            comparison_id = ctx.state_store.add_comparison(
                session_id=active_session_id,
                prompt_text=cleaned_prompt,
                left_image_uri=left_uri,
                right_image_uri=right_uri,
                winner=winner_value,
                critique=critique,
                outcome=outcome,
            )
            ctx.state_store.mark_rollout_feedback_complete(active_rollout_id.strip(), comparison_id)
            if active_job_id.strip():
                rollout = next(
                    (
                        item
                        for item in ctx.state_store.list_rollouts_for_job(active_job_id.strip(), limit=300)
                        if item["id"] == active_rollout_id.strip()
                    ),
                    None,
                )
                if rollout is not None and rollout.get("rollout_type") == "candidate_comparison":
                    left_candidate_id = rollout.get("left_candidate_id")
                    right_candidate_id = rollout.get("right_candidate_id")
                    judgement = judge_critique(
                        original_prompt=cleaned_prompt,
                        human_winner=winner_value,
                        left={
                            "candidate_id": left_candidate_id,
                            "role": "left",
                            "system_prompt": left_system_prompt,
                        },
                        right={
                            "candidate_id": right_candidate_id,
                            "role": "right",
                            "system_prompt": right_system_prompt,
                        },
                        critique=critique,
                    )
                    if outcome == "winner" and winner_value == "left":
                        ctx.state_store.update_candidate_feedback_stats(
                            winner_candidate_id=left_candidate_id,
                            loser_candidate_id=right_candidate_id,
                            winner_margin=float(judgement["winner_margin"]),
                            critique_confidence=float(judgement["critique_confidence"]),
                            judge_metadata=judgement,
                        )
                    elif outcome == "winner" and winner_value == "right":
                        ctx.state_store.update_candidate_feedback_stats(
                            winner_candidate_id=right_candidate_id,
                            loser_candidate_id=left_candidate_id,
                            winner_margin=float(judgement["winner_margin"]),
                            critique_confidence=float(judgement["critique_confidence"]),
                            judge_metadata=judgement,
                        )
                    else:
                        ctx.state_store.update_candidate_feedback_stats(
                            winner_candidate_id=None,
                            loser_candidate_id=None,
                            tied_candidate_ids=[left_candidate_id, right_candidate_id],
                            winner_margin=0.0,
                            critique_confidence=float(judgement["critique_confidence"]),
                            judge_metadata=judgement,
                        )
                    ctx.state_store.recompute_gepa_frontier_for_job(active_job_id.strip())
            completed_count = "0"
            if active_job_id.strip():
                completed_count = str(ctx.state_store.count_completed_rollouts_for_job(active_job_id.strip()))
            return (
                f"Score saved in session `{active_session_id}`. Prepare a new matchup for the next rollout.",
                active_session_id,
                "",
                completed_count,
                "",
                "",
                "",
                "",
                "baseline_candidate",
                None,
                None,
                "",
                "",
                "",
                "",
                "",
            )

        refresh_jobs_btn.click(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        )
        refresh_inspector_jobs_btn.click(
            _refresh_inspector_jobs,
            outputs=[inspector_job, workflow_status],
        )
        app.load(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        ).then(
            _refresh_inspector_jobs,
            outputs=[inspector_job, workflow_status],
        )
        create_job_btn.click(
            _create_job,
            inputs=[
                create_job_name,
                create_job_description,
                create_job_seed_prompt,
                create_job_category,
                create_job_threshold,
            ],
            outputs=[selected_job, workflow_status],
        )
        update_job_btn.click(
            _update_selected_job,
            inputs=[
                active_job_id_state,
                update_job_name,
                update_job_description,
                update_job_category,
                update_job_threshold,
            ],
            outputs=[workflow_status],
        ).then(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        )
        archive_job_btn.click(
            _archive_selected_job,
            inputs=[active_job_id_state],
            outputs=[active_job_id_state, workflow_status],
        ).then(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        )
        use_selected_job_btn.click(
            _use_selected_job,
            inputs=selected_job,
            outputs=[
                active_job_id_state,
                active_job_name,
                compiled_prompt_view,
                baseline_prompt_text,
                candidate_prompt_text,
                workflow_status,
                rollout_workflow_group,
                completed_feedback_count,
            ],
        ).then(
            _gepa_button_state,
            inputs=[active_job_id_state],
            outputs=[run_gepa_btn, workflow_status, latest_gepa_run_status, latest_gepa_run_id_state, gepa_poll_timer],
        ).then(
            lambda: "",
            outputs=[active_rollout_id_state],
        )

        sample_prompt_btn.click(
            _sample_prompt,
            inputs=[active_job_id_state],
            outputs=[
                sampled_prompt,
                prompt_category_state,
                prompt_selection_mode_state,
                prompt_llm_score_state,
                prompt_llm_reason_state,
                workflow_status,
            ],
        )

        prepare_matchup_btn.click(
            _prepare_training_matchup,
            inputs=[sampled_prompt, active_job_id_state],
            outputs=[
                train_left_candidate_id_state,
                train_right_candidate_id_state,
                train_left_system_prompt_state,
                train_right_system_prompt_state,
                baseline_candidate_label,
                candidate_candidate_label,
                baseline_prompt_text,
                candidate_prompt_text,
                train_rollout_type_state,
                baseline_image,
                candidate_image,
                baseline_path_state,
                candidate_path_state,
                active_rollout_id_state,
                workflow_status,
            ],
        )

        generate_pair_btn.click(
            _generate_training_pair,
            inputs=[
                sampled_prompt,
                active_job_id_state,
                train_left_candidate_id_state,
                train_right_candidate_id_state,
                train_left_system_prompt_state,
                train_right_system_prompt_state,
                train_rollout_type_state,
                prompt_category_state,
                prompt_selection_mode_state,
                prompt_llm_score_state,
                prompt_llm_reason_state,
            ],
            outputs=[
                baseline_image,
                candidate_image,
                baseline_path_state,
                candidate_path_state,
                active_rollout_id_state,
                train_left_system_prompt_state,
                baseline_prompt_text,
                candidate_prompt_text,
                workflow_status,
            ],
        )

        submit_score_btn.click(
            _submit_score,
            inputs=[
                sampled_prompt,
                baseline_path_state,
                candidate_path_state,
                winner_choice,
                critique_text,
                session_id_state,
                active_rollout_id_state,
                active_job_id_state,
                train_left_candidate_id_state,
                train_right_candidate_id_state,
                train_left_system_prompt_state,
                train_right_system_prompt_state,
                train_rollout_type_state,
                baseline_candidate_label,
                candidate_candidate_label,
            ],
            outputs=[
                workflow_status,
                session_id_state,
                active_rollout_id_state,
                completed_feedback_count,
                train_left_candidate_id_state,
                train_right_candidate_id_state,
                train_left_system_prompt_state,
                train_right_system_prompt_state,
                train_rollout_type_state,
                baseline_image,
                candidate_image,
                baseline_path_state,
                candidate_path_state,
                baseline_candidate_label,
                candidate_candidate_label,
                critique_text,
            ],
        ).then(
            _gepa_button_state,
            inputs=[active_job_id_state],
            outputs=[run_gepa_btn, workflow_status, latest_gepa_run_status, latest_gepa_run_id_state, gepa_poll_timer],
        )

        run_gepa_btn.click(
            _run_gepa_optimization,
            inputs=[active_job_id_state, minibatch_size],
            outputs=[
                latest_gepa_run_status,
                completed_feedback_count,
                workflow_status,
                latest_gepa_run_id_state,
                gepa_poll_timer,
            ],
        )

        archive_pending_candidates_btn.click(
            _archive_pending_candidates,
            inputs=[active_job_id_state],
            outputs=[
                workflow_status,
                latest_gepa_run_status,
                completed_feedback_count,
                latest_gepa_run_id_state,
                gepa_poll_timer,
            ],
        ).then(
            _gepa_button_state,
            inputs=[active_job_id_state],
            outputs=[run_gepa_btn, workflow_status, latest_gepa_run_status, latest_gepa_run_id_state, gepa_poll_timer],
        )

        promote_frontier_btn.click(
            _promote_best_frontier_candidate,
            inputs=[active_job_id_state],
            outputs=[compiled_prompt_view, baseline_prompt_text, candidate_prompt_text, workflow_status],
        )

        refresh_gepa_status_btn.click(
            _refresh_gepa_status,
            inputs=[latest_gepa_run_id_state, active_job_id_state],
            outputs=[
                latest_gepa_run_status,
                completed_feedback_count,
                workflow_status,
                compiled_prompt_view,
                baseline_prompt_text,
                candidate_prompt_text,
                gepa_poll_timer,
            ],
        )
        show_gepa_logs_btn.click(
            _show_gepa_logs,
            inputs=[latest_gepa_run_id_state],
            outputs=[gepa_run_logs, workflow_status],
        )
        gepa_poll_timer.tick(
            _refresh_gepa_status,
            inputs=[latest_gepa_run_id_state, active_job_id_state],
            outputs=[
                latest_gepa_run_status,
                completed_feedback_count,
                workflow_status,
                compiled_prompt_view,
                baseline_prompt_text,
                candidate_prompt_text,
                gepa_poll_timer,
            ],
        )

        generate_latest_check_btn.click(
            _generate_latest_prompt_check,
            inputs=[latest_check_prompt, active_job_id_state],
            outputs=[
                latest_check_baseline_image,
                latest_check_candidate_image,
                latest_check_baseline_path_state,
                latest_check_candidate_path_state,
                latest_check_baseline_prompt,
                latest_check_candidate_prompt,
                workflow_status,
            ],
        )

        refresh_rollouts_btn.click(
            _load_rollout_choices,
            inputs=[inspector_job],
            outputs=[inspector_rollout, workflow_status],
        )
        inspector_rollout.change(
            _show_rollout_details,
            inputs=[inspector_rollout, inspector_job],
            outputs=[inspector_baseline_image, inspector_candidate_image, rollout_metadata, workflow_status],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()

