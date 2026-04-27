from __future__ import annotations

import random
from pathlib import Path

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

        with gr.Row():
            sample_prompt_btn = gr.Button("Sample Prompt")
            generate_baseline_btn = gr.Button("Generate Baseline")
            regenerate_btn = gr.Button("Regenerate")

        sampled_prompt = gr.Textbox(label="Sampled Prompt", interactive=False)
        reprompt_text = gr.Textbox(label="Reprompt", placeholder="Edit reprompt before regenerate")

        with gr.Row():
            baseline_image = gr.Image(label="Baseline", type="filepath")
            regenerated_image = gr.Image(label="Regenerated", type="filepath")

        winner_choice = gr.Radio(
            choices=["baseline", "regenerated", "tie"],
            value="tie",
            label="Winner",
        )
        critique_text = gr.Textbox(label="Critique (optional)")
        submit_score_btn = gr.Button("Submit Score")
        workflow_status = gr.Markdown("Ready. Start with `Sample Prompt`.")

        def _sample_prompt() -> tuple[str, str, str]:
            prompts = sample_prompts_from_local_source(
                prompt_source_root=DEFAULT_PROMPT_SOURCE_ROOT,
                candidate_count=DEFAULT_PROMPT_CANDIDATE_COUNT,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            selected = random.choice(prompts).strip()
            return selected, selected, "Prompt sampled. Click `Generate Baseline`."

        def _generate_baseline(prompt: str) -> tuple[str | None, str, str]:
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                return None, "", "Sample a prompt first."

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

        def _regenerate(prompt: str, reprompt: str, baseline_path: str) -> tuple[str | None, str, str]:
            cleaned_prompt = prompt.strip()
            cleaned_reprompt = reprompt.strip()
            if not cleaned_prompt:
                return None, "", "Sample a prompt first."
            if not baseline_path.strip():
                return None, "", "Generate baseline first."
            if not cleaned_reprompt:
                return None, "", "Reprompt is required."

            baseline_image_path = Path(baseline_path)
            if not baseline_image_path.exists():
                return None, "", "Baseline image file is missing. Regenerate baseline."

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
            return str(regenerated_path), str(regenerated_path), "Regenerated image ready. Pick winner and submit score."

        def _submit_score(
            prompt: str,
            baseline_path: str,
            regenerated_path: str,
            winner: str,
            critique: str,
            session_id: str,
        ) -> tuple[str, str]:
            cleaned_prompt = prompt.strip()
            left_uri = baseline_path.strip()
            right_uri = regenerated_path.strip()
            if not cleaned_prompt:
                return "Prompt is required before scoring.", session_id
            if not left_uri or not right_uri:
                return "Generate both baseline and regenerated images before scoring.", session_id
            if left_uri == right_uri:
                return "Baseline and regenerated image URIs must differ.", session_id

            active_session_id = session_id.strip()
            if not active_session_id:
                active_session_id = ctx.state_store.create_rating_session("ui-workflow-session")

            if winner == "baseline":
                winner_value = "left"
                outcome = "winner"
            elif winner == "regenerated":
                winner_value = "right"
                outcome = "winner"
            else:
                winner_value = None
                outcome = "cant_decide"

            ctx.state_store.add_comparison(
                session_id=active_session_id,
                prompt_text=cleaned_prompt,
                left_image_uri=left_uri,
                right_image_uri=right_uri,
                winner=winner_value,
                critique=critique,
                outcome=outcome,
            )
            return f"Score saved in session `{active_session_id}`.", active_session_id

        sample_prompt_btn.click(
            _sample_prompt,
            outputs=[sampled_prompt, reprompt_text, workflow_status],
        ).then(
            lambda sampled: sampled,
            inputs=sampled_prompt,
            outputs=prompt_state,
        )

        generate_baseline_btn.click(
            _generate_baseline,
            inputs=sampled_prompt,
            outputs=[baseline_image, baseline_path_state, workflow_status],
        )

        regenerate_btn.click(
            _regenerate,
            inputs=[sampled_prompt, reprompt_text, baseline_path_state],
            outputs=[regenerated_image, regenerated_path_state, workflow_status],
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
            ],
            outputs=[workflow_status, session_id_state],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()

