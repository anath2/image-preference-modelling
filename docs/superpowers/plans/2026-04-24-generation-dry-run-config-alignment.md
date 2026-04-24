# Generation Dry Run Config Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the online generation dry run to load the real image-generation config from `config.py` so it uses `IMAGE_MODEL`, `OPENROUTER_API_KEY`, and optional `OPENROUTER_BASE_URL` instead of a dry-run-only config contract.

**Architecture:** Add a shared `ImageGenerationModelSettings` dataclass beside `PromptRewriteModelSettings`, then make `generation_pipeline.py` consume that shared config while keeping the Hugging Face prompt source and timeout as internal defaults. Cover the refactor with focused unit tests and keep the online integration test pointed at the real dry-run entry point with no manual settings injection.

**Tech Stack:** Python 3.13, pytest, python-dotenv, requests, OpenRouter, Hugging Face dataset viewer

---

## File Map

- Modify: `src/image_preference_modelling/config.py`
  - Add `ImageGenerationModelSettings` that reads `IMAGE_MODEL`, `OPENROUTER_API_KEY`, and optional `OPENROUTER_BASE_URL`.
- Modify: `src/image_preference_modelling/generation_pipeline.py`
  - Remove `GenerationDryRunSettings`.
  - Import and use `ImageGenerationModelSettings`.
  - Keep Hugging Face dataset/config/split/column and timeout as module-level defaults.
- Create: `tests/test_config.py`
  - Verify the shared image-generation config loader reads the real env contract and raises helpful errors.
- Create: `tests/test_generation_pipeline.py`
  - Verify `run_generation_dry_run()` resolves shared config from `config.py` when no config object is passed.
- Modify: `tests/test_generation_pipeline_online.py`
  - Call `run_generation_dry_run(output_dir=Path("data"))` directly, then verify file creation and cleanup.
- Reuse unchanged: `tests/conftest.py`
  - The `--online` gate already exists and should continue to control the integration test.

### Task 1: Add Shared Image Generation Config

**Files:**
- Create: `tests/test_config.py`
- Modify: `src/image_preference_modelling/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from __future__ import annotations

import pytest

from image_preference_modelling.config import ImageGenerationModelSettings


def test_image_generation_model_settings_from_env_reads_existing_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_MODEL", "openai/gpt-5.4-image-2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    settings = ImageGenerationModelSettings.from_env()

    assert settings == ImageGenerationModelSettings(
        image_model="openai/gpt-5.4-image-2",
        openrouter_api_key="secret",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )


def test_image_generation_model_settings_from_env_requires_image_model_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IMAGE_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="IMAGE_MODEL, OPENROUTER_API_KEY"):
        ImageGenerationModelSettings.from_env()
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`

Expected: `FAIL` with an import error because `ImageGenerationModelSettings` does not exist in `src/image_preference_modelling/config.py` yet.

- [ ] **Step 3: Add the shared config loader in `src/image_preference_modelling/config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class PromptRewriteModelSettings:
    prompt_model: str
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "PromptRewriteModelSettings":
        load_dotenv(override=False)
        prompt_model = os.getenv("PROMPT_MODEL", "").strip()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"

        missing: list[str] = []
        if not prompt_model:
            missing.append("PROMPT_MODEL")
        if not api_key:
            missing.append("OPENROUTER_API_KEY")

        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(f"Missing required environment variables for prompt rewrite: {missing_joined}")

        return cls(
            prompt_model=prompt_model,
            openrouter_api_key=api_key,
            openrouter_base_url=base_url.rstrip("/"),
        )


@dataclass(frozen=True)
class ImageGenerationModelSettings:
    image_model: str
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "ImageGenerationModelSettings":
        load_dotenv(override=False)
        image_model = os.getenv("IMAGE_MODEL", "").strip()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"

        missing: list[str] = []
        if not image_model:
            missing.append("IMAGE_MODEL")
        if not api_key:
            missing.append("OPENROUTER_API_KEY")

        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables for image generation: {missing_joined}"
            )

        return cls(
            image_model=image_model,
            openrouter_api_key=api_key,
            openrouter_base_url=base_url.rstrip("/"),
        )
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`

Expected: `2 passed`

- [ ] **Step 5: Commit the shared config change**

```bash
git add src/image_preference_modelling/config.py tests/test_config.py
git commit -m "refactor: add shared image generation config"
```

### Task 2: Refactor `generation_pipeline.py` To Use Shared Config

**Files:**
- Create: `tests/test_generation_pipeline.py`
- Modify: `src/image_preference_modelling/generation_pipeline.py`
- Test: `tests/test_generation_pipeline.py`

- [ ] **Step 1: Write the failing pipeline unit test**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from image_preference_modelling.config import ImageGenerationModelSettings
from image_preference_modelling.generation_pipeline import run_generation_dry_run


def test_run_generation_dry_run_loads_shared_image_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_settings = ImageGenerationModelSettings(
        image_model="openai/gpt-5.4-image-2",
        openrouter_api_key="secret",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        ImageGenerationModelSettings,
        "from_env",
        classmethod(lambda cls: resolved_settings),
    )
    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.sample_prompt_from_huggingface",
        lambda timeout_seconds=60.0: "test prompt",
    )

    def _fake_generate_image(
        prompt: str,
        settings: ImageGenerationModelSettings,
        *,
        timeout_seconds: float = 60.0,
    ) -> str:
        calls["prompt"] = prompt
        calls["settings"] = settings
        calls["timeout_seconds"] = timeout_seconds
        return "data:image/png;base64,aGVsbG8="

    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.generate_image_from_openrouter",
        _fake_generate_image,
    )

    result = run_generation_dry_run(output_dir=tmp_path / "data")

    assert calls["prompt"] == "test prompt"
    assert calls["settings"] == resolved_settings
    assert calls["timeout_seconds"] == 60.0
    assert result.prompt == "test prompt"
    assert result.image_path.exists()
    assert result.image_path.parent == tmp_path / "data"
```

- [ ] **Step 2: Run the pipeline unit test to verify it fails**

Run: `uv run pytest tests/test_generation_pipeline.py::test_run_generation_dry_run_loads_shared_image_settings -v`

Expected: `FAIL` because `generation_pipeline.py` still depends on `GenerationDryRunSettings` and does not resolve `ImageGenerationModelSettings.from_env()`.

- [ ] **Step 3: Refactor `src/image_preference_modelling/generation_pipeline.py`**

```python
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from image_preference_modelling.config import ImageGenerationModelSettings

DEFAULT_HF_PROMPT_DATASET = "daspartho/stable-diffusion-prompts"
DEFAULT_HF_PROMPT_CONFIG = "default"
DEFAULT_HF_PROMPT_SPLIT = "train"
DEFAULT_HF_PROMPT_COLUMN = "prompt"
DEFAULT_TIMEOUT_SECONDS = 60.0


class PromptSourceClientError(RuntimeError):
    """Raised when the prompt source request fails."""


class ImageGenerationClientError(RuntimeError):
    """Raised when the image generation request fails."""


class GenerationDryRunOutputError(ValueError):
    """Raised when an external response is missing required fields."""


@dataclass(frozen=True)
class GenerationDryRunResult:
    prompt: str
    image_path: Path


def sample_prompt_from_huggingface(
    *,
    dataset: str = DEFAULT_HF_PROMPT_DATASET,
    config: str = DEFAULT_HF_PROMPT_CONFIG,
    split: str = DEFAULT_HF_PROMPT_SPLIT,
    prompt_column: str = DEFAULT_HF_PROMPT_COLUMN,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    try:
        response = requests.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": dataset, "config": config, "split": split},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PromptSourceClientError(f"Prompt source request failed: {exc}") from exc

    return _extract_prompt(response.json(), prompt_column=prompt_column)


def generate_image_from_openrouter(
    prompt: str,
    settings: ImageGenerationModelSettings,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.image_model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
    }
    try:
        response = requests.post(
            f"{settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ImageGenerationClientError(f"Image generation request failed: {exc}") from exc

    return _extract_image_data_url(response.json())


def run_generation_dry_run(
    output_dir: Path,
    *,
    settings: ImageGenerationModelSettings | None = None,
) -> GenerationDryRunResult:
    active_settings = settings or ImageGenerationModelSettings.from_env()
    prompt = sample_prompt_from_huggingface(timeout_seconds=DEFAULT_TIMEOUT_SECONDS)
    image_data_url = generate_image_from_openrouter(
        prompt,
        active_settings,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    image_path = save_generated_image(image_data_url, output_dir)
    return GenerationDryRunResult(prompt=prompt, image_path=image_path)
```

- [ ] **Step 4: Run the pipeline-focused tests to verify they pass**

Run: `uv run pytest tests/test_generation_pipeline.py tests/test_config.py tests/test_intent_rewriter.py -v`

Expected: `8 passed`

- [ ] **Step 5: Commit the pipeline refactor**

```bash
git add src/image_preference_modelling/generation_pipeline.py tests/test_generation_pipeline.py tests/test_config.py
git commit -m "refactor: use shared config for generation dry run"
```

### Task 3: Update The Online Integration Test To Exercise The Real Entry Point

**Files:**
- Modify: `tests/test_generation_pipeline_online.py`
- Test: `tests/test_generation_pipeline_online.py`

- [ ] **Step 1: Rewrite the online test to stop constructing a dry-run settings object**

```python
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

    result = generation_pipeline.run_generation_dry_run(output_dir=data_dir)

    assert result.prompt
    assert result.image_path.exists()
    assert result.image_path.is_file()
    assert result.image_path.parent == data_dir
    assert result.image_path.stat().st_size > 0

    result.image_path.unlink()

    assert not result.image_path.exists()
```

- [ ] **Step 2: Run the online test without `--online` to verify the skip path**

Run: `uv run pytest tests/test_generation_pipeline_online.py -q`

Expected: `1 skipped`

- [ ] **Step 3: Run the online test with `--online` to verify the real config path**

Run: `uv run pytest --online tests/test_generation_pipeline_online.py -q`

Expected: `1 passed`

- [ ] **Step 4: Run the full suite to verify no regressions**

Run: `uv run pytest -q`

Expected: `19 passed, 1 skipped`

- [ ] **Step 5: Commit the integration-test update**

```bash
git add tests/test_generation_pipeline_online.py
git commit -m "test: align online dry run with shared config"
```
