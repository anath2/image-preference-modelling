from __future__ import annotations

import base64
import binascii
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from image_preference_modelling.config import ImageGenerationModelSettings


class PromptSourceClientError(RuntimeError):
    """Raised when the prompt source request fails."""


class ImageGenerationClientError(RuntimeError):
    """Raised when the image generation request fails."""


class GenerationDryRunOutputError(ValueError):
    """Raised when an external response is missing required fields."""


DEFAULT_HF_PROMPT_DATASET = "succinctly/midjourney-prompts"
DEFAULT_HF_PROMPT_CONFIG = "default"
DEFAULT_HF_PROMPT_SPLIT = "train"
DEFAULT_HF_PROMPT_COLUMN = "text"
DEFAULT_PROMPT_SOURCE_ROOT = Path("data/prompt_sources")
DEFAULT_PROMPT_CANDIDATE_COUNT = 20
DEFAULT_TIMEOUT_SECONDS = 60.0
_ALPHA_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")



@dataclass(frozen=True)
class GenerationDryRunResult:
    prompt: str
    image_path: Path


def ensure_prompt_source_parquet(
    *,
    dataset: str = DEFAULT_HF_PROMPT_DATASET,
    config: str = DEFAULT_HF_PROMPT_CONFIG,
    split: str = DEFAULT_HF_PROMPT_SPLIT,
    prompt_column: str = DEFAULT_HF_PROMPT_COLUMN,
    prompt_source_root: Path = DEFAULT_PROMPT_SOURCE_ROOT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    parquet_path = prompt_source_root / _prompt_dataset_dirname(dataset) / config / f"{split}.parquet"
    if parquet_path.exists():
        try:
            _validate_prompt_source_parquet(parquet_path, prompt_column=prompt_column)
            return parquet_path
        except GenerationDryRunOutputError:
            parquet_path.unlink(missing_ok=True)

    parquet_url = _discover_prompt_source_parquet_url(
        dataset=dataset,
        config=config,
        split=split,
        timeout_seconds=timeout_seconds,
    )

    try:
        response = requests.get(parquet_url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PromptSourceClientError(f"Prompt source parquet download failed: {exc}") from exc

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = parquet_path.with_name(f".{parquet_path.name}.tmp")
    temp_path.write_bytes(response.content)
    try:
        _validate_prompt_source_parquet(temp_path, prompt_column=prompt_column)
    except GenerationDryRunOutputError:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(parquet_path)
    return parquet_path


def read_prompts_from_parquet(parquet_path: Path, *, prompt_column: str = DEFAULT_HF_PROMPT_COLUMN) -> list[str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(parquet_path, columns=[prompt_column])
        column = table.column(prompt_column).to_pylist()
    except (KeyError, OSError, pa.ArrowException) as exc:
        raise GenerationDryRunOutputError(
            f"Could not read prompt source parquet `{parquet_path}`"
        ) from exc
    return [prompt for prompt in column if isinstance(prompt, str)]


def sample_prompts_from_local_source(
    *,
    prompt_source_root: Path = DEFAULT_PROMPT_SOURCE_ROOT,
    candidate_count: int = DEFAULT_PROMPT_CANDIDATE_COUNT,
    dataset: str = DEFAULT_HF_PROMPT_DATASET,
    config: str = DEFAULT_HF_PROMPT_CONFIG,
    split: str = DEFAULT_HF_PROMPT_SPLIT,
    prompt_column: str = DEFAULT_HF_PROMPT_COLUMN,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[str]:
    parquet_path = ensure_prompt_source_parquet(
        dataset=dataset,
        config=config,
        split=split,
        prompt_column=prompt_column,
        prompt_source_root=prompt_source_root,
        timeout_seconds=timeout_seconds,
    )
    prompts = [
        prompt
        for prompt in read_prompts_from_parquet(parquet_path, prompt_column=prompt_column)
        if _is_usable_prompt(prompt)
    ]
    if not prompts:
        raise GenerationDryRunOutputError("Prompt source parquet did not contain usable prompts")
    if len(prompts) <= candidate_count:
        return prompts
    return random.sample(prompts, k=candidate_count)


def _discover_prompt_source_parquet_url(
    *,
    dataset: str,
    config: str,
    split: str,
    timeout_seconds: float,
) -> str:
    try:
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": dataset},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PromptSourceClientError(f"Prompt source metadata request failed: {exc}") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise GenerationDryRunOutputError("Prompt source metadata response was malformed")

    parquet_files = payload.get("parquet_files")
    if not isinstance(parquet_files, list):
        raise GenerationDryRunOutputError("Prompt source metadata missing parquet_files")

    for parquet_file in parquet_files:
        if not isinstance(parquet_file, dict):
            continue
        if parquet_file.get("config") != config or parquet_file.get("split") != split:
            continue
        parquet_url = parquet_file.get("url")
        if isinstance(parquet_url, str) and parquet_url:
            return parquet_url

    raise GenerationDryRunOutputError(
        f"Prompt source metadata did not contain parquet for config={config!r} split={split!r}"
    )


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
        "modalities": ["image"],
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


def save_generated_image(data_url: str, output_dir: Path) -> Path:
    image_suffix, image_bytes = _decode_image_data_url(data_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"online-dry-run{image_suffix}"
    image_path.write_bytes(image_bytes)
    return image_path


def run_generation_dry_run(
    output_dir: Path,
    *,
    settings: ImageGenerationModelSettings | None = None,
) -> GenerationDryRunResult:
    active_settings = settings or ImageGenerationModelSettings.from_env()
    prompts = sample_prompts_from_local_source(
        prompt_source_root=DEFAULT_PROMPT_SOURCE_ROOT,
        candidate_count=DEFAULT_PROMPT_CANDIDATE_COUNT,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    last_failure: ImageGenerationClientError | GenerationDryRunOutputError | None = None

    for prompt in prompts:
        try:
            image_data_url = generate_image_from_openrouter(
                prompt,
                active_settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            image_path = save_generated_image(image_data_url, output_dir)
            return GenerationDryRunResult(prompt=prompt, image_path=image_path)
        except (ImageGenerationClientError, GenerationDryRunOutputError) as exc:
            last_failure = exc

    if last_failure is not None:
        raise last_failure

    raise GenerationDryRunOutputError("Prompt source did not return any candidate prompts")


def _extract_prompt(payload: dict[str, Any], *, prompt_column: str) -> str:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GenerationDryRunOutputError("Prompt source response missing rows")

    for item in rows:
        if not isinstance(item, dict):
            continue
        row = item.get("row")
        if not isinstance(row, dict):
            continue
        prompt = row.get(prompt_column)
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    raise GenerationDryRunOutputError(
        f"Prompt source response did not contain a non-empty `{prompt_column}` value"
    )


def _prompt_dataset_dirname(dataset: str) -> str:
    return dataset.replace("/", "_")


def _validate_prompt_source_parquet(
    parquet_path: Path,
    *,
    prompt_column: str,
) -> None:
    read_prompts_from_parquet(parquet_path, prompt_column=prompt_column)


def _is_usable_prompt(prompt: str) -> bool:
    stripped_prompt = prompt.strip()
    alpha_tokens = _ALPHA_TOKEN_PATTERN.findall(stripped_prompt)
    return len(stripped_prompt) >= 40 and len(alpha_tokens) >= 5


def _extract_image_data_url(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GenerationDryRunOutputError("OpenRouter response missing choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise GenerationDryRunOutputError("OpenRouter response missing message")

    images = message.get("images")
    if not isinstance(images, list) or not images:
        raise GenerationDryRunOutputError("OpenRouter response missing images")

    image = images[0]
    if not isinstance(image, dict):
        raise GenerationDryRunOutputError("OpenRouter response image payload is malformed")

    image_url = image.get("image_url")
    if not isinstance(image_url, dict):
        raise GenerationDryRunOutputError("OpenRouter response missing image_url payload")

    data_url = image_url.get("url")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise GenerationDryRunOutputError("OpenRouter response missing image data URL")

    return data_url.strip()


def _decode_image_data_url(data_url: str) -> tuple[str, bytes]:
    header, separator, encoded = data_url.partition(",")
    if separator != "," or ";base64" not in header:
        raise GenerationDryRunOutputError("Image data URL must be base64 encoded")

    mime_type = header.removeprefix("data:").split(";", maxsplit=1)[0]
    if not mime_type.startswith("image/"):
        raise GenerationDryRunOutputError("Image data URL must have an image MIME type")

    extension_by_mime = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }
    image_suffix = extension_by_mime.get(mime_type)
    if image_suffix is None:
        subtype = mime_type.removeprefix("image/").strip()
        if not subtype:
            raise GenerationDryRunOutputError("Unsupported image MIME type")
        image_suffix = f".{subtype}"

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise GenerationDryRunOutputError("Image data URL payload is not valid base64") from exc

    if not image_bytes:
        raise GenerationDryRunOutputError("Decoded image payload was empty")

    return image_suffix, image_bytes
