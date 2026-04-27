from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _load_openrouter_config(model_env_var: str, *, context_label: str) -> tuple[str, str, str]:
    load_dotenv(override=False)
    model_value = os.getenv(model_env_var, "").strip()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"

    missing: list[str] = []
    if not model_value:
        missing.append(model_env_var)
    if not api_key:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(f"Missing required environment variables for {context_label}: {missing_joined}")

    return model_value, api_key, base_url.rstrip("/")


@dataclass(frozen=True)
class PromptRewriteModelSettings:
    prompt_model: str
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "PromptRewriteModelSettings":
        # Load local .env for developer runs without overriding exported env vars.
        prompt_model, api_key, base_url = _load_openrouter_config(
            "PROMPT_MODEL", context_label="prompt rewrite"
        )

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
        image_model, api_key, base_url = _load_openrouter_config(
            "IMAGE_MODEL", context_label="image generation"
        )

        return cls(
            image_model=image_model,
            openrouter_api_key=api_key,
            openrouter_base_url=base_url.rstrip("/"),
        )
