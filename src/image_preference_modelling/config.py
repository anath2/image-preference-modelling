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
        # Load local .env for developer runs without overriding exported env vars.
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
