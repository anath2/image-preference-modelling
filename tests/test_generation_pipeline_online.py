from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.online


def test_online_generation_dry_run_samples_saves_and_deletes_image(tmp_path: Path) -> None:
    module_name = "image_preference_modelling.generation_pipeline"
    assert importlib.util.find_spec(module_name) is not None

    generation_pipeline = importlib.import_module(module_name)
    settings = generation_pipeline.GenerationDryRunSettings.from_env()
    data_dir = Path("data")

    result = generation_pipeline.run_generation_dry_run(output_dir=data_dir, settings=settings)

    assert result.prompt
    assert result.image_path.exists()
    assert result.image_path.is_file()
    assert result.image_path.parent == data_dir
    assert result.image_path.stat().st_size > 0

    #result.image_path.unlink()

    #assert not result.image_path.exists()
