from pathlib import Path

import gradio as gr

from image_preference_modelling.app_context import AppContext
from image_preference_modelling.gradio_app import build_app
from image_preference_modelling.jobs.job_launcher import JobLauncher
from image_preference_modelling.storage.state_store import StateStore


def _build_context(tmp_path: Path) -> AppContext:
    store = StateStore(db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    return AppContext(state_store=store, job_launcher=JobLauncher(store))


def test_build_app_uses_plain_code_component_for_run_logs(tmp_path: Path) -> None:
    app = build_app(_build_context(tmp_path))

    run_log_output = next(
        block
        for block in app.blocks.values()
        if isinstance(block, gr.Code) and block.label == "Run log"
    )

    assert run_log_output.language is None
    assert run_log_output.interactive is False
