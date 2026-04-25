from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.online


def test_online_generation_dry_run_samples_saves_and_deletes_image() -> None:
    module_name = "image_preference_modelling.generation_pipeline"
    assert importlib.util.find_spec(module_name) is not None

    generation_pipeline = importlib.import_module(module_name)
    data_dir = Path("data")
    stale_image_paths = list(data_dir.glob("online-dry-run.*"))

    for stale_image_path in stale_image_paths:
        stale_image_path.unlink()

    assert not list(data_dir.glob("online-dry-run.*"))

    result = generation_pipeline.run_generation_dry_run(output_dir=data_dir)

    try:
        assert result.prompt
        assert result.image_path.exists()
        assert result.image_path.is_file()
        assert result.image_path.parent == data_dir
        assert result.image_path.stat().st_size > 0
    finally:
        for image_path in data_dir.glob("online-dry-run.*"):
            image_path.unlink()

    assert not result.image_path.exists()
