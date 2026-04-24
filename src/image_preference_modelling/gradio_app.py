from __future__ import annotations

from typing import Any

import gradio as gr

from image_preference_modelling.app_context import AppContext, default_context


def _table_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No records yet._"

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def build_app(context: AppContext | None = None) -> gr.Blocks:
    ctx = context or default_context()

    with gr.Blocks(title="Gradio Operator Cockpit") as app:
        gr.Markdown("# Gradio Operator Cockpit")
        gr.Markdown(
            "Control plane for prompt-generation runs, review sessions, reward-model tracking, and GEPA/evaluation workflows."
        )

        with gr.Tab("Overview"):
            overview_output = gr.JSON(label="Current metrics")
            refresh_overview = gr.Button("Refresh overview")
            refresh_overview.click(lambda: ctx.state_store.overview_metrics(), outputs=overview_output)

        with gr.Tab("Prompt Sets"):
            gr.Markdown(
                "\n".join(
                    [
                        "### Prompt Set Registry",
                        "- Ingestion and cleaning pipelines are tracked as run configs in `Runs`.",
                        "- Coverage and split diagnostics can be attached as artifacts to prompt-set preparation runs.",
                    ]
                )
            )

        with gr.Tab("Runs"):
            with gr.Row():
                run_type = gr.Dropdown(
                    choices=["generation", "reward_model", "gepa", "evaluation"],
                    value="generation",
                    label="Run type",
                )
                display_name = gr.Textbox(label="Display name", value="New run")
            run_config = gr.Code(
                label="Run config JSON",
                language="json",
                value='{\n  "note": "update with run parameters"\n}',
            )
            launch_run = gr.Button("Create run")
            start_run = gr.Button("Dispatch selected run")
            cancel_run = gr.Button("Cancel selected run")
            selected_run_id = gr.Textbox(label="Run ID")
            run_status = gr.Markdown()
            run_list = gr.Markdown(label="Recent runs")
            run_log_output = gr.Code(label="Run log", interactive=False)
            refresh_runs = gr.Button("Refresh runs")
            refresh_run_log = gr.Button("Refresh selected run log")

            def _create_run(run_type_value: str, name_value: str, config_value: str) -> tuple[str, str]:
                config: dict[str, Any]
                try:
                    import json

                    config = json.loads(config_value)
                except Exception as exc:  # noqa: BLE001 - displayed in UI
                    return "", f"Invalid JSON config: {exc}"

                run_id = ctx.state_store.create_run(
                    run_type=run_type_value,
                    display_name=name_value,
                    config=config,
                )
                rows = ctx.state_store.list_runs()
                return run_id, _table_markdown(
                    rows,
                    ["id", "run_type", "display_name", "status", "created_at", "finished_at"],
                )

            def _start_run(run_id: str) -> str:
                if not run_id.strip():
                    return "Provide a run id first."
                try:
                    message = ctx.job_launcher.dispatch_run(run_id.strip())
                except ValueError as exc:
                    return str(exc)
                return message

            def _cancel_run(run_id: str) -> str:
                if not run_id.strip():
                    return "Provide a run id first."
                try:
                    return ctx.job_launcher.cancel_run(run_id.strip())
                except ValueError as exc:
                    return str(exc)

            def _read_run_log(run_id: str) -> str:
                if not run_id.strip():
                    return "Select a run id to load logs."
                run = ctx.state_store.get_run(run_id.strip())
                if run is None:
                    return f"Run {run_id.strip()} not found."
                events = ctx.state_store.list_run_events(run_id.strip())
                if not events:
                    return "No run events yet."
                return "\n".join(
                    f"{event['created_at']} [{event['level']}] {event['message']}" for event in events
                )

            launch_run.click(_create_run, inputs=[run_type, display_name, run_config], outputs=[selected_run_id, run_list])
            start_run.click(_start_run, inputs=selected_run_id, outputs=run_status)
            cancel_run.click(_cancel_run, inputs=selected_run_id, outputs=run_status)
            refresh_runs.click(
                lambda: _table_markdown(
                    ctx.state_store.list_runs(),
                    ["id", "run_type", "display_name", "status", "created_at", "finished_at"],
                ),
                outputs=run_list,
            )
            refresh_run_log.click(_read_run_log, inputs=selected_run_id, outputs=run_log_output)

        with gr.Tab("Review Queue"):
            session_name = gr.Textbox(label="Session name", value="bootstrap-session")
            create_session = gr.Button("Create rating session")
            session_id_output = gr.Textbox(label="Session ID")

            prompt_text = gr.Textbox(label="Prompt text")
            left_uri = gr.Textbox(label="Left image URI")
            right_uri = gr.Textbox(label="Right image URI")
            winner = gr.Radio(
                choices=["left", "right", "none"],
                label="Winner",
                value="none",
            )
            outcome = gr.Dropdown(
                choices=["winner", "both_good", "both_bad", "cant_decide"],
                value="winner",
                label="Outcome",
            )
            critique = gr.Textbox(label="One-line critique")
            submit_comparison = gr.Button("Save comparison")
            comparison_status = gr.Markdown()
            review_hint = gr.Markdown("Submit a comparison to reveal the next queue hint.")
            recent_comparisons = gr.Markdown(label="Recent comparisons")
            refresh_comparisons = gr.Button("Refresh comparisons")

            create_session.click(
                lambda name: ctx.state_store.create_rating_session(name),
                inputs=session_name,
                outputs=session_id_output,
            )

            def _submit_comparison(
                session_id: str,
                prompt: str,
                left: str,
                right: str,
                winner_choice: str,
                critique_text: str,
                outcome_choice: str,
            ) -> str:
                if not session_id.strip():
                    return "Create or provide a rating session id first."
                if not prompt.strip():
                    return "Prompt text is required."
                if not left.strip() or not right.strip():
                    return "Both left and right image URIs are required."
                if left.strip() == right.strip():
                    return "Left and right image URIs must differ."
                winner_value = None if winner_choice == "none" else winner_choice
                if outcome_choice == "winner" and winner_value is None:
                    return "Outcome `winner` requires choosing left or right."
                ctx.state_store.add_comparison(
                    session_id=session_id.strip(),
                    prompt_text=prompt,
                    left_image_uri=left,
                    right_image_uri=right,
                    winner=winner_value,
                    critique=critique_text,
                    outcome=outcome_choice,
                )
                return "Comparison saved."

            def _next_review_hint() -> str:
                rows = ctx.state_store.list_recent_comparisons(limit=1)
                if not rows:
                    return "No comparisons logged yet. Start with a seeded prompt pair."
                last = rows[0]
                if last["outcome"] == "cant_decide":
                    return "Last item was undecided. Queue a clarifying prompt variant next."
                return f"Last outcome: `{last['outcome']}`. Continue with the next unresolved prompt pair."

            submit_comparison.click(
                _submit_comparison,
                inputs=[session_id_output, prompt_text, left_uri, right_uri, winner, critique, outcome],
                outputs=comparison_status,
            )
            submit_comparison.click(_next_review_hint, outputs=review_hint)
            refresh_comparisons.click(
                lambda: _table_markdown(
                    ctx.state_store.list_recent_comparisons(),
                    ["id", "rating_session_id", "winner", "outcome", "created_at", "critique"],
                ),
                outputs=recent_comparisons,
            )

        with gr.Tab("Reward Model"):
            gr.Markdown(
                "\n".join(
                    [
                        "### Reward Model Registry",
                        "- Use `Runs` with `run_type=reward_model` to create and track training jobs.",
                        "- Attach agreement metrics and checkpoints to each run artifact directory.",
                    ]
                )
            )

        with gr.Tab("Rewriter / GEPA"):
            gr.Markdown(
                "\n".join(
                    [
                        "### GEPA Workspace",
                        "- Launch GEPA iterations from `Runs` with `run_type=gepa`.",
                        "- Store candidate prompt variants and frontier summaries as run artifacts.",
                    ]
                )
            )

        with gr.Tab("Evaluation"):
            gr.Markdown(
                "\n".join(
                    [
                        "### Blind Evaluation",
                        "- Launch eval batches from `Runs` with `run_type=evaluation`.",
                        "- Compare baseline vs promoted rewriter using blind pairwise results in `Review Queue`.",
                    ]
                )
            )

    return app


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()

