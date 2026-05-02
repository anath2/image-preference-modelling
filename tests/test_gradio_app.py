from pathlib import Path

import gradio as gr
import pytest

from image_preference_modelling.app_context import AppContext
from image_preference_modelling.gradio_app import (
    _build_gepa_run_config,
    _resolve_active_system_prompt,
    _winner_to_storage_outcome,
    build_app,
)
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
        seed_system_prompt="seed policy",
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
        if isinstance(block, (gr.Textbox, gr.Image, gr.Radio, gr.Dropdown, gr.Number))
    }
    winner_radio = next(
        block for block in app.blocks.values() if isinstance(block, gr.Radio) and block.label == "Winner"
    )

    assert "Sample Prompt" in buttons
    assert "Prepare Matchup" in buttons
    assert "Generate Left / Right Images" in buttons
    assert "Submit Score" in buttons
    assert "Create Job" in buttons
    assert "Update Selected Job" in buttons
    assert "Archive Selected Job" in buttons
    assert "Refresh Inspector Jobs" in buttons
    assert "Load Rollouts" in buttons
    assert "Use Selected Job" in buttons
    assert "Refresh Jobs" in buttons
    assert "Generate Mutation Now" in buttons
    assert "Archive Pending Candidates" in buttons
    assert "Promote Best Frontier Candidate" in buttons
    assert "Refresh Mutation Status" in buttons
    assert "Show Mutation Logs" in buttons
    assert "Generate Best Candidate Check" in buttons
    assert "Active Job" in labels
    assert "Selected Job Name" in labels
    assert "Latest System Prompt" in labels
    assert "New Job Name" in labels
    assert "Seed System Prompt" in labels
    assert "Guided Sampling Category" in labels
    assert "GEPA Enable Threshold" in labels
    assert "Completed Feedback Count" in labels
    assert "Mutation Feedback Batch Size" in labels
    assert "Latest Mutation Run Status" in labels
    assert "Mutation Run Logs" in labels
    assert "Best Candidate Check Prompt" in labels
    assert "Best Check Baseline System Prompt" in labels
    assert "Best Check Candidate System Prompt" in labels
    assert "Best Check Baseline" in labels
    assert "Best Check Candidate" in labels
    assert "Prompt" in labels
    assert "Left Candidate" in labels
    assert "Right Candidate" in labels
    assert "Left System Prompt" in labels
    assert "Right System Prompt" in labels
    assert "Inspect Job" in labels
    assert "Rollout" in labels
    assert "Rollout Metadata" in labels
    assert "Inspector Baseline" in labels
    assert "Inspector Candidate" in labels
    assert "Left" in labels
    assert "Right" in labels
    assert tuple(choice[0] for choice in winner_radio.choices) == (
        "left",
        "right",
        "no_clear_winner",
    )


@pytest.mark.parametrize(
    ("winner", "expected"),
    [
        ("baseline", ("left", "winner")),
        ("candidate", ("right", "winner")),
        ("left", ("left", "winner")),
        ("right", ("right", "winner")),
        ("no_clear_winner", (None, "no_clear_winner")),
        ("both_good", (None, "no_clear_winner")),
        ("both_bad", (None, "no_clear_winner")),
        ("cant_decide", (None, "no_clear_winner")),
    ],
)
def test_winner_to_storage_outcome_mappings(winner: str, expected: tuple[str | None, str]) -> None:
    assert _winner_to_storage_outcome(winner) == expected


def test_build_gepa_run_config_includes_job_and_minibatch() -> None:
    config = _build_gepa_run_config(
        job_id="job_123",
        minibatch_size=3,
        selected_rollout_ids=["rollout_1", "rollout_2", "rollout_3"],
        active_candidate_id="candidate_a",
        compiled_prompt="compiled policy",
    )
    assert config["job_id"] == "job_123"
    assert config["minibatch_size"] == 3
    assert config["selected_rollout_ids"] == ["rollout_1", "rollout_2", "rollout_3"]
    assert config["optimizer_backend"] == "prompt_mutation"


def test_resolve_active_system_prompt_prefers_compiled_prompt() -> None:
    assert (
        _resolve_active_system_prompt(
            {
                "seed_system_prompt": "seed prompt",
                "latest_system_prompt": "latest prompt",
                "compiled_system_prompt": "compiled prompt",
            }
        )
        == "latest prompt"
    )
    assert (
        _resolve_active_system_prompt(
            {
                "seed_system_prompt": "seed prompt",
                "latest_system_prompt": "",
                "compiled_system_prompt": "",
            }
        )
        == "seed prompt"
    )
