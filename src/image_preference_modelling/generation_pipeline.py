from __future__ import annotations

import base64
import binascii
import random
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from image_preference_modelling.config import ImageGenerationModelSettings


class GenerationDryRunOutputError(ValueError):
    """Raised when an external response is missing required fields."""


DEFAULT_HF_PROMPT_DATASET = "succinctly/midjourney-prompts"
DEFAULT_HF_PROMPT_CONFIG = "default"
DEFAULT_HF_PROMPT_SPLIT = "train"
DEFAULT_HF_PROMPT_COLUMN = "text"
DEFAULT_PROMPT_SOURCE_ROOT = Path("data/prompt_sources")
DEFAULT_PROMPT_CANDIDATE_COUNT = 20
DEFAULT_TIMEOUT_SECONDS = 120.0
_ALPHA_TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")
_IMAGE_SUFFIX_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}



@dataclass(frozen=True)
class GenerationDryRunResult:
    prompt: str
    baseline_image_path: Path
    candidate_image_path: Path
    # Compatibility alias for legacy callers expecting a single output image.
    image_path: Path


def ensure_prompt_source_parquet(
    *,
    dataset: str = DEFAULT_HF_PROMPT_DATASET,
    config: str = DEFAULT_HF_PROMPT_CONFIG,
    split: str = DEFAULT_HF_PROMPT_SPLIT,
    prompt_source_root: Path = DEFAULT_PROMPT_SOURCE_ROOT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    parquet_path = prompt_source_root / _prompt_dataset_dirname(dataset) / config / f"{split}.parquet"
    if parquet_path.exists():
        return parquet_path

    parquet_url = _discover_prompt_source_parquet_url(
        dataset=dataset,
        config=config,
        split=split,
        timeout_seconds=timeout_seconds,
    )

    response = requests.get(parquet_url, timeout=timeout_seconds)
    response.raise_for_status()

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_bytes(response.content)
    return parquet_path


def read_prompts_from_parquet(parquet_path: Path, *, prompt_column: str = DEFAULT_HF_PROMPT_COLUMN) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[prompt_column])
    column = table.column(prompt_column).to_pylist()
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
    response = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset},
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    payload = response.json()
    for parquet_file in payload["parquet_files"]:
        if parquet_file.get("config") != config or parquet_file.get("split") != split:
            continue
        return parquet_file["url"]

    raise LookupError(f"Prompt source has no parquet for config={config!r} split={split!r}")


def generate_image_from_openrouter(
    prompt: str,
    settings: ImageGenerationModelSettings,
    *,
    system_prompt: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    messages: list[dict[str, object]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return _post_openrouter_image_completion(
        settings=settings,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )


def generate_candidate_image_from_openrouter(
    original_prompt: str,
    system_prompt: str,
    settings: ImageGenerationModelSettings,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    return generate_image_from_openrouter(
        original_prompt,
        settings=settings,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
    )


def build_candidate_system_prompt(*, original_prompt: str, regeneration_instructions: str) -> str:
    return regeneration_instructions.strip()


def _post_openrouter_image_completion(
    *,
    settings: ImageGenerationModelSettings,
    messages: list[dict[str, object]],
    timeout_seconds: float,
) -> str:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.image_model,
        "messages": messages,
        "modalities": ["image"],
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    response = requests.post(
        f"{settings.openrouter_base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    return _extract_image_data_url(response.json())


def save_generated_image(data_url: str, output_dir: Path, *, stem: str = "online-dry-run") -> Path:
    image_suffix, image_bytes = _decode_image_data_url(data_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}{image_suffix}"
    image_path.write_bytes(image_bytes)
    return image_path


def run_generation_dry_run(
    output_dir: Path,
    *,
    settings: ImageGenerationModelSettings | None = None,
) -> GenerationDryRunResult:
    active_settings = settings or ImageGenerationModelSettings.from_env()
    refinement_instruction = (
        "Refine this image to better match the visual intent while preserving subject and composition. "
        "Improve style, coherence, and detail without changing the core scene."
    )
    prompts = sample_prompts_from_local_source(
        prompt_source_root=DEFAULT_PROMPT_SOURCE_ROOT,
        candidate_count=DEFAULT_PROMPT_CANDIDATE_COUNT,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    )
    last_failure: GenerationDryRunOutputError | None = None

    for prompt in prompts:
        try:
            baseline_data_url = generate_image_from_openrouter(
                prompt,
                active_settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            baseline_image_path = save_generated_image(
                baseline_data_url,
                output_dir,
                stem="online-dry-run-baseline",
            )
            candidate_system_prompt = build_candidate_system_prompt(
                original_prompt=prompt,
                regeneration_instructions=refinement_instruction,
            )
            candidate_data_url = generate_candidate_image_from_openrouter(
                prompt,
                candidate_system_prompt,
                active_settings,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            )
            candidate_image_path = save_generated_image(
                candidate_data_url,
                output_dir,
                stem="online-dry-run-candidate",
            )
            return GenerationDryRunResult(
                prompt=prompt,
                baseline_image_path=baseline_image_path,
                candidate_image_path=candidate_image_path,
                image_path=candidate_image_path,
            )
        except GenerationDryRunOutputError as exc:
            last_failure = exc

    if last_failure is not None:
        raise last_failure

    raise GenerationDryRunOutputError("Prompt source did not return any candidate prompts")


def _prompt_dataset_dirname(dataset: str) -> str:
    return dataset.replace("/", "_")


def _is_usable_prompt(prompt: str) -> bool:
    stripped_prompt = prompt.strip()
    alpha_tokens = _ALPHA_TOKEN_PATTERN.findall(stripped_prompt)
    return len(stripped_prompt) >= 40 and len(alpha_tokens) >= 5


def _extract_image_data_url(payload: dict) -> str:
    try:
        data_url = payload["choices"][0]["message"]["images"][0]["image_url"]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GenerationDryRunOutputError("OpenRouter response missing image data URL") from exc

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

    image_suffix = _IMAGE_SUFFIX_BY_MIME.get(mime_type)
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
