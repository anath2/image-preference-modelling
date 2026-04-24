from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


class PromptSourceClientError(RuntimeError):
    """Raised when the prompt source request fails."""


class ImageGenerationClientError(RuntimeError):
    """Raised when the image generation request fails."""


class GenerationDryRunOutputError(ValueError):
    """Raised when an external response is missing required fields."""


@dataclass(frozen=True)
class GenerationDryRunSettings:
    openrouter_api_key: str
    openrouter_image_model: str = "google/gemini-2.5-flash-image"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    hf_prompt_dataset: str = "daspartho/stable-diffusion-prompts"
    hf_prompt_config: str = "default"
    hf_prompt_split: str = "train"
    hf_prompt_column: str = "prompt"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "GenerationDryRunSettings":
        # Load local .env for developer runs without overriding exported env vars.
        load_dotenv(override=False)

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        image_model = os.getenv("OPENROUTER_IMAGE_MODEL", "").strip() or "google/gemini-2.5-flash-image"
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"
        hf_prompt_dataset = os.getenv("HF_PROMPT_DATASET", "").strip() or "daspartho/stable-diffusion-prompts"
        hf_prompt_config = os.getenv("HF_PROMPT_CONFIG", "").strip() or "default"
        hf_prompt_split = os.getenv("HF_PROMPT_SPLIT", "").strip() or "train"
        hf_prompt_column = os.getenv("HF_PROMPT_COLUMN", "").strip() or "prompt"
        timeout_raw = os.getenv("GENERATION_DRY_RUN_TIMEOUT_SECONDS", "").strip()

        missing: list[str] = []
        if not api_key:
            missing.append("OPENROUTER_API_KEY")

        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(f"Missing required environment variables for generation dry run: {missing_joined}")

        timeout_seconds = 60.0
        if timeout_raw:
            try:
                timeout_seconds = float(timeout_raw)
            except ValueError as exc:
                raise ValueError("GENERATION_DRY_RUN_TIMEOUT_SECONDS must be a float") from exc

        return cls(
            openrouter_api_key=api_key,
            openrouter_image_model=image_model,
            openrouter_base_url=base_url.rstrip("/"),
            hf_prompt_dataset=hf_prompt_dataset,
            hf_prompt_config=hf_prompt_config,
            hf_prompt_split=hf_prompt_split,
            hf_prompt_column=hf_prompt_column,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class GenerationDryRunResult:
    prompt: str
    image_path: Path


def sample_prompt_from_huggingface(settings: GenerationDryRunSettings) -> str:
    try:
        response = requests.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={
                "dataset": settings.hf_prompt_dataset,
                "config": settings.hf_prompt_config,
                "split": settings.hf_prompt_split,
            },
            timeout=settings.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PromptSourceClientError(f"Prompt source request failed: {exc}") from exc

    return _extract_prompt(response.json(), prompt_column=settings.hf_prompt_column)


def generate_image_from_openrouter(prompt: str, settings: GenerationDryRunSettings) -> str:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openrouter_image_model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
    }

    try:
        response = requests.post(
            f"{settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=settings.timeout_seconds,
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
    settings: GenerationDryRunSettings | None = None,
) -> GenerationDryRunResult:
    active_settings = settings or GenerationDryRunSettings.from_env()
    prompt = sample_prompt_from_huggingface(active_settings)
    image_data_url = generate_image_from_openrouter(prompt, active_settings)
    image_path = save_generated_image(image_data_url, output_dir)
    return GenerationDryRunResult(prompt=prompt, image_path=image_path)


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
