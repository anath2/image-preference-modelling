from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import gradio as gr

from image_preference_modelling.app_context import AppContext, default_context
from image_preference_modelling.config import ImageGenerationModelSettings
from image_preference_modelling.generation_pipeline import (
    DEFAULT_PROMPT_SOURCE_ROOT,
    DEFAULT_PROMPT_CANDIDATE_COUNT,
    DEFAULT_TIMEOUT_SECONDS,
    generate_image_from_openrouter,
    generate_image_refinement_from_openrouter,
    image_file_to_data_url,
    sample_prompts_from_local_source,
    save_generated_image,
)


def _workflow_output_dir() -> Path:
    output_dir = Path(".local") / "artifacts" / "ui_workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _winner_to_storage_outcome(winner: str) -> tuple[str | None, str]:
    if winner == "baseline":
        return "left", "winner"
    if winner == "regenerated":
        return "right", "winner"
    if winner in {"both_good", "both_bad", "cant_decide"}:
        return None, winner
    raise ValueError(f"Unsupported winner value: {winner}")


def _job_choices(jobs: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(f"{job['name']} ({job['id']})", job["id"]) for job in jobs]


def _resolve_active_refinement_prompt(job: dict[str, Any]) -> str:
    return (job.get("compiled_gepa_prompt") or job["seed_refinement_prompt"]).strip()


def _build_gepa_run_config(
    *,
    job_id: str,
    minibatch_size: int,
    selected_rollout_ids: list[str],
    active_candidate_id: str | None,
    compiled_prompt: str | None,
) -> dict[str, Any]:
    dspy_model = os.getenv("DSPY_MODEL", "").strip() or os.getenv("PROMPT_MODEL", "").strip()
    return {
        "job_id": job_id,
        "minibatch_size": minibatch_size,
        "selected_rollout_ids": selected_rollout_ids,
        "active_candidate_id": active_candidate_id,
        "compiled_prompt": compiled_prompt,
        "optimizer_backend": "dspy_gepa",
        "dspy_model": dspy_model or None,
    }


def build_app(context: AppContext | None = None) -> gr.Blocks:
    ctx = context or default_context()
    output_dir = _workflow_output_dir()

    with gr.Blocks(title="Gradio Operator Cockpit") as app:
        gr.Markdown("# Image Preference Workflow")
        gr.Markdown(
            "Sample a prompt, generate baseline image, edit reprompt, regenerate, then score the winner."
        )

        prompt_state = gr.State(value="")
        baseline_path_state = gr.State(value="")
        regenerated_path_state = gr.State(value="")
        session_id_state = gr.State(value="")
        active_job_id_state = gr.State(value="")
        active_rollout_id_state = gr.State(value="")
        active_refinement_prompt_state = gr.State(value="")
        latest_gepa_run_id_state = gr.State(value="")

        gr.Markdown("## Aesthetic Job")
        with gr.Row():
            selected_job = gr.Dropdown(label="Active Job", choices=[], value=None)
            refresh_jobs_btn = gr.Button("Refresh Jobs")
            use_selected_job_btn = gr.Button("Use Selected Job")
        create_job_name = gr.Textbox(label="New Job Name")
        create_job_description = gr.Textbox(label="New Job Description")
        create_job_seed_prompt = gr.Textbox(label="Seed Refinement Prompt")
        create_job_btn = gr.Button("Create Job")
        active_job_name = gr.Textbox(label="Selected Job Name", interactive=False)
        compiled_prompt_view = gr.Textbox(label="Compiled GEPA Prompt", interactive=False)

        with gr.Group(visible=False) as rollout_workflow_group:
            with gr.Row():
                sample_prompt_btn = gr.Button("Sample Prompt")
                generate_baseline_btn = gr.Button("Generate Baseline")
                regenerate_btn = gr.Button("Regenerate")

            sampled_prompt = gr.Textbox(label="Sampled Prompt", interactive=False)
            reprompt_text = gr.Textbox(label="Active Refinement Prompt", interactive=False)

            with gr.Row():
                baseline_image = gr.Image(label="Baseline", type="filepath")
                regenerated_image = gr.Image(label="Regenerated", type="filepath")

            winner_choice = gr.Radio(
                choices=["baseline", "regenerated", "both_good", "both_bad", "cant_decide"],
                value="cant_decide",
                label="Winner",
            )
            critique_text = gr.Textbox(label="Critique")
            submit_score_btn = gr.Button("Submit Score")

        gr.Markdown("## GEPA Optimization")
        completed_feedback_count = gr.Textbox(label="Completed Feedback Count", interactive=False, value="0")
        minibatch_size = gr.Number(
            label="GEPA Minibatch Size",
            value=3,
            minimum=1,
            maximum=10,
            precision=0,
        )
        run_gepa_btn = gr.Button("Run GEPA Optimization")
        refresh_gepa_status_btn = gr.Button("Refresh GEPA Status")
        show_gepa_logs_btn = gr.Button("Show GEPA Run Logs")
        gepa_poll_timer = gr.Timer(value=2.0, active=False)
        latest_gepa_run_status = gr.Textbox(label="Latest GEPA Run Status", interactive=False, value="No GEPA run yet.")
        gepa_run_logs = gr.Textbox(
            label="GEPA Run Logs",
            interactive=False,
            lines=14,
            max_lines=24,
            value="No GEPA logs yet.",
        )
        workflow_status = gr.Markdown("Ready. Start with `Sample Prompt`.")

        def _refresh_job_choices() -> tuple[gr.Dropdown, str]:
            jobs = ctx.state_store.list_aesthetic_jobs()
            return gr.update(choices=_job_choices(jobs)), "Job list refreshed."

        def _create_job(name: str, description: str, seed_prompt: str) -> tuple[gr.Dropdown, str]:
            if not name.strip() or not seed_prompt.strip():
                return gr.update(), "Job name and seed refinement prompt are required."
            ctx.state_store.create_aesthetic_job(name, description, seed_prompt)
            jobs = ctx.state_store.list_aesthetic_jobs()
            return gr.update(choices=_job_choices(jobs)), "Aesthetic job created."

        def _use_selected_job(job_id: str | None) -> tuple[str, str, str, str, Any, str]:
            selected = (job_id or "").strip()
            if not selected:
                return "", "", "", "Select a job first.", gr.update(visible=False), "0"
            job = ctx.state_store.get_aesthetic_job(selected)
            if job is None or job["status"] != "active":
                return "", "", "", "Selected job is unavailable.", gr.update(visible=False), "0"
            active_prompt = _resolve_active_refinement_prompt(job)
            completed_count = ctx.state_store.count_completed_rollouts_for_job(selected)
            return (
                selected,
                job["name"],
                active_prompt,
                f"Using aesthetic job `{job['name']}`.",
                gr.update(visible=True),
                str(completed_count),
            )

        def _run_gepa_optimization(
            active_job_id: str,
            minibatch_value: int | float,
        ) -> tuple[str, str, str, str, Any]:
            selected_job_id = active_job_id.strip()
            if not selected_job_id:
                return "No GEPA run yet.", "0", "Select and activate an aesthetic job first.", "", gr.update(active=False)
            job = ctx.state_store.get_aesthetic_job(selected_job_id)
            if job is None:
                return "No GEPA run yet.", "0", "Selected job is unavailable.", "", gr.update(active=False)
            chosen_minibatch = int(minibatch_value)
            if chosen_minibatch < 1 or chosen_minibatch > 10:
                return (
                    "No GEPA run yet.",
                    str(ctx.state_store.count_completed_rollouts_for_job(selected_job_id)),
                    "Minibatch size must be between 1 and 10.",
                    "",
                    gr.update(active=False),
                )
            completed_count = ctx.state_store.count_completed_rollouts_for_job(selected_job_id)
            if completed_count < chosen_minibatch:
                return (
                    "No GEPA run yet.",
                    str(completed_count),
                    (
                        f"Need at least {chosen_minibatch} completed rollouts; currently {completed_count}."
                    ),
                    "",
                    gr.update(active=False),
                )
            selected_rollout_ids = ctx.state_store.list_completed_rollout_ids_for_job(
                selected_job_id,
                limit=chosen_minibatch,
            )
            run_config = _build_gepa_run_config(
                job_id=selected_job_id,
                minibatch_size=chosen_minibatch,
                selected_rollout_ids=selected_rollout_ids,
                active_candidate_id=job.get("active_candidate_id"),
                compiled_prompt=job.get("compiled_gepa_prompt"),
            )
            run_id = ctx.state_store.create_run(
                run_type="gepa",
                display_name=f"GEPA Optimization ({job['name']})",
                config=run_config,
            )
            dispatch_message = ctx.job_launcher.dispatch_run(run_id)
            run = ctx.state_store.get_run(run_id)
            status = run["status"] if run is not None else "queued"
            return (
                f"{run_id} ({status})",
                str(completed_count),
                f"GEPA optimization started. {dispatch_message}",
                run_id,
                gr.update(active=True),
            )

        def _refresh_gepa_status(
            run_id: str,
            active_job_id: str,
        ) -> tuple[str, str, str, str, str, Any]:
            active_job = active_job_id.strip()
            if not run_id.strip():
                return "No GEPA run yet.", "0", "No GEPA run has been started yet.", "", "", gr.update(active=False)
            run = ctx.state_store.get_run(run_id.strip())
            if run is None:
                return "No GEPA run yet.", "0", "Latest GEPA run no longer exists.", "", "", gr.update(active=False)
            status = run["status"]
            status_text = f"{run['id']} ({status})"
            completed_count = "0"
            compiled_prompt = ""
            refinement_prompt = ""
            if active_job:
                job = ctx.state_store.get_aesthetic_job(active_job)
                if job is not None:
                    completed_count = str(ctx.state_store.count_completed_rollouts_for_job(active_job))
                    compiled_prompt = str(job.get("compiled_gepa_prompt") or "")
                    refinement_prompt = _resolve_active_refinement_prompt(job)
            if status == "completed":
                message = "GEPA run completed. Active job prompt refreshed."
                timer_update = gr.update(active=False)
            elif status == "failed":
                events = ctx.state_store.list_run_events(run["id"], limit=20)
                error_event = next((e for e in reversed(events) if e["level"] == "ERROR"), None)
                error_text = error_event["message"] if error_event else "No error details available."
                message = f"GEPA run failed: {error_text}"
                timer_update = gr.update(active=False)
            elif status == "cancelled":
                message = "GEPA run was cancelled."
                timer_update = gr.update(active=False)
            else:
                message = f"GEPA run is {status}. Refresh again shortly."
                timer_update = gr.update(active=True)
            return status_text, completed_count, message, compiled_prompt, refinement_prompt, timer_update

        def _show_gepa_logs(run_id: str) -> tuple[str, str]:
            selected_run = run_id.strip()
            if not selected_run:
                return "No GEPA logs yet.", "No GEPA run has been started yet."
            run = ctx.state_store.get_run(selected_run)
            if run is None:
                return "No GEPA logs yet.", "Latest GEPA run no longer exists."
            events = ctx.state_store.list_run_events(selected_run, limit=300)
            if not events:
                return "No GEPA logs yet.", "No run events are available yet."
            lines = [
                f"{event['created_at']} [{event['level']}] {event['message']}"
                for event in events
            ]
            return "\n".join(lines), f"Showing logs for `{selected_run}`."

        def _sample_prompt() -> tuple[str, str]:
            prompts = sample_prompts_from_local_source(
                prompt_source_root=DEFAULT_PROMPT_SOURCE_ROOT,
                candidate_count=DEFAULT_PROMPT_CANDIDATE_COUNT,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            selected = random.choice(prompts).strip()
            return selected, "Prompt sampled. Click `Generate Baseline`."

        def _generate_baseline(prompt: str, active_job_id: str) -> tuple[str | None, str, str]:
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                return None, "", "Sample a prompt first."
            if not active_job_id.strip():
                return None, "", "Select and activate an aesthetic job first."

            settings = ImageGenerationModelSettings.from_env()
            baseline_data_url = generate_image_from_openrouter(
                cleaned_prompt,
                settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            baseline_path = save_generated_image(
                baseline_data_url,
                output_dir,
                stem="ui-baseline",
            )
            return str(baseline_path), str(baseline_path), "Baseline generated. Edit reprompt and click `Regenerate`."

        def _regenerate(
            prompt: str,
            reprompt: str,
            baseline_path: str,
            active_job_id: str,
        ) -> tuple[str | None, str, str, str]:
            cleaned_prompt = prompt.strip()
            cleaned_reprompt = reprompt.strip()
            if not cleaned_prompt:
                return None, "", "", "Sample a prompt first."
            if not baseline_path.strip():
                return None, "", "", "Generate baseline first."
            if not active_job_id.strip():
                return None, "", "", "Select and activate an aesthetic job first."
            if not cleaned_reprompt:
                return None, "", "", "Refinement prompt is required."

            baseline_image_path = Path(baseline_path)
            if not baseline_image_path.exists():
                return None, "", "", "Baseline image file is missing. Regenerate baseline."

            settings = ImageGenerationModelSettings.from_env()
            source_image_data_url = image_file_to_data_url(baseline_image_path)
            refined_data_url = generate_image_refinement_from_openrouter(
                f"{cleaned_reprompt}\n\nOriginal prompt:\n{cleaned_prompt}",
                source_image_data_url,
                settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            regenerated_path = save_generated_image(
                refined_data_url,
                output_dir,
                stem="ui-regenerated",
            )
            rollout_id = ctx.state_store.create_rollout(
                job_id=active_job_id.strip(),
                prompt_text=cleaned_prompt,
                intent_text=cleaned_prompt,
                baseline_image_uri=str(baseline_image_path),
                refined_image_uri=str(regenerated_path),
                candidate_id=None,
                refinement_prompt=cleaned_reprompt,
                model_config={"image_model": settings.image_model},
            )
            return (
                str(regenerated_path),
                str(regenerated_path),
                rollout_id,
                "Regenerated image ready. Pick winner and submit score.",
            )

        def _submit_score(
            prompt: str,
            baseline_path: str,
            regenerated_path: str,
            winner: str,
            critique: str,
            session_id: str,
            active_rollout_id: str,
            active_job_id: str,
        ) -> tuple[str, str, str, str]:
            cleaned_prompt = prompt.strip()
            left_uri = baseline_path.strip()
            right_uri = regenerated_path.strip()
            if not cleaned_prompt:
                return "Prompt is required before scoring.", session_id, active_rollout_id, "0"
            if not left_uri or not right_uri:
                return (
                    "Generate both baseline and regenerated images before scoring.",
                    session_id,
                    active_rollout_id,
                    "0",
                )
            if left_uri == right_uri:
                return "Baseline and regenerated image URIs must differ.", session_id, active_rollout_id, "0"
            if not active_rollout_id.strip():
                return "Regenerate an image to create a rollout before scoring.", session_id, active_rollout_id, "0"
            if not critique.strip():
                return "Critique is required before scoring.", session_id, active_rollout_id, "0"

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
            completed_count = "0"
            if active_job_id.strip():
                completed_count = str(ctx.state_store.count_completed_rollouts_for_job(active_job_id.strip()))
            return f"Score saved in session `{active_session_id}`.", active_session_id, "", completed_count

        refresh_jobs_btn.click(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        )
        app.load(
            _refresh_job_choices,
            outputs=[selected_job, workflow_status],
        )
        create_job_btn.click(
            _create_job,
            inputs=[create_job_name, create_job_description, create_job_seed_prompt],
            outputs=[selected_job, workflow_status],
        )
        use_selected_job_btn.click(
            _use_selected_job,
            inputs=selected_job,
            outputs=[
                active_job_id_state,
                active_job_name,
                compiled_prompt_view,
                workflow_status,
                rollout_workflow_group,
                completed_feedback_count,
            ],
        ).then(
            lambda prompt: prompt,
            inputs=compiled_prompt_view,
            outputs=[reprompt_text],
        ).then(
            lambda: "",
            outputs=[active_rollout_id_state],
        )

        sample_prompt_btn.click(
            _sample_prompt,
            outputs=[sampled_prompt, workflow_status],
        ).then(
            lambda sampled: sampled,
            inputs=sampled_prompt,
            outputs=prompt_state,
        )

        generate_baseline_btn.click(
            _generate_baseline,
            inputs=[sampled_prompt, active_job_id_state],
            outputs=[baseline_image, baseline_path_state, workflow_status],
        )

        regenerate_btn.click(
            _regenerate,
            inputs=[sampled_prompt, reprompt_text, baseline_path_state, active_job_id_state],
            outputs=[regenerated_image, regenerated_path_state, active_rollout_id_state, workflow_status],
        )

        submit_score_btn.click(
            _submit_score,
            inputs=[
                prompt_state,
                baseline_path_state,
                regenerated_path_state,
                winner_choice,
                critique_text,
                session_id_state,
                active_rollout_id_state,
                active_job_id_state,
            ],
            outputs=[workflow_status, session_id_state, active_rollout_id_state, completed_feedback_count],
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

        refresh_gepa_status_btn.click(
            _refresh_gepa_status,
            inputs=[latest_gepa_run_id_state, active_job_id_state],
            outputs=[
                latest_gepa_run_status,
                completed_feedback_count,
                workflow_status,
                compiled_prompt_view,
                reprompt_text,
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
                reprompt_text,
                gepa_poll_timer,
            ],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()

