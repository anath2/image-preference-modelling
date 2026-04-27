from pathlib import Path

import gradio as gr

from image_preference_modelling.app_context import AppContext
from image_preference_modelling.gradio_app import build_app
from image_preference_modelling.jobs.job_launcher import JobLauncher
from image_preference_modelling.storage.state_store import StateStore


def _build_context(tmp_path: Path) -> AppContext:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    return AppContext(state_store=store, job_launcher=JobLauncher(store))


def test_build_app_exposes_single_flow_controls(tmp_path: Path) -> None:
    app = build_app(_build_context(tmp_path))

    buttons = {
        block.value
        for block in app.blocks.values()
        if isinstance(block, gr.Button) and isinstance(block.value, str)
    }
    labels = {
        block.label
        for block in app.blocks.values()
        if isinstance(block, (gr.Textbox, gr.Image, gr.Radio))
    }
    winner_radio = next(
        block for block in app.blocks.values() if isinstance(block, gr.Radio) and block.label == "Winner"
    )

    assert "Sample Prompt" in buttons
    assert "Generate Baseline" in buttons
    assert "Regenerate" in buttons
    assert "Submit Score" in buttons
    assert "Sampled Prompt" in labels
    assert "Reprompt" in labels
    assert "Baseline" in labels
    assert "Regenerated" in labels
    assert tuple(choice[0] for choice in winner_radio.choices) == ("baseline", "regenerated", "tie")
