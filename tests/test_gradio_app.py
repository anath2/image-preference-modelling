from pathlib import Path

import gradio as gr
import pytest

from image_preference_modelling.app_context import AppContext
from image_preference_modelling.gradio_app import _winner_to_storage_outcome, build_app
from image_preference_modelling.jobs.job_launcher import JobLauncher
from image_preference_modelling.storage.state_store import StateStore


def _build_context(tmp_path: Path) -> AppContext:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    return AppContext(state_store=store, job_launcher=JobLauncher(store))


def test_build_app_exposes_single_flow_controls(tmp_path: Path) -> None:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    store.create_aesthetic_job(
        name="test-job",
        description="test",
        seed_refinement_prompt="seed policy",
    )
    app = build_app(AppContext(state_store=store, job_launcher=JobLauncher(store)))

    buttons = {
        block.value
        for block in app.blocks.values()
        if isinstance(block, gr.Button) and isinstance(block.value, str)
    }
    labels = {
        block.label
        for block in app.blocks.values()
        if isinstance(block, (gr.Textbox, gr.Image, gr.Radio, gr.Dropdown))
    }
    winner_radio = next(
        block for block in app.blocks.values() if isinstance(block, gr.Radio) and block.label == "Winner"
    )

    assert "Sample Prompt" in buttons
    assert "Generate Baseline" in buttons
    assert "Regenerate" in buttons
    assert "Submit Score" in buttons
    assert "Create Job" in buttons
    assert "Use Selected Job" in buttons
    assert "Refresh Jobs" in buttons
    assert "Active Job" in labels
    assert "Selected Job Name" in labels
    assert "Compiled GEPA Prompt" in labels
    assert "New Job Name" in labels
    assert "Seed Refinement Prompt" in labels
    assert "Sampled Prompt" in labels
    assert "Active Refinement Prompt" in labels
    assert "Baseline" in labels
    assert "Regenerated" in labels
    assert tuple(choice[0] for choice in winner_radio.choices) == (
        "baseline",
        "regenerated",
        "both_good",
        "both_bad",
        "cant_decide",
    )


@pytest.mark.parametrize(
    ("winner", "expected"),
    [
        ("baseline", ("left", "winner")),
        ("regenerated", ("right", "winner")),
        ("both_good", (None, "both_good")),
        ("both_bad", (None, "both_bad")),
        ("cant_decide", (None, "cant_decide")),
    ],
)
def test_winner_to_storage_outcome_mappings(winner: str, expected: tuple[str | None, str]) -> None:
    assert _winner_to_storage_outcome(winner) == expected
