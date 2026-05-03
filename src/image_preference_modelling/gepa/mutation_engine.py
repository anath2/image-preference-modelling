from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from image_preference_modelling.config import PromptRewriteModelSettings
from image_preference_modelling.generation_pipeline import DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class PromptMutation:
    compiled_prompt: str
    metadata: dict[str, Any]
    backend: str


_MUTATION_SYSTEM_PROMPT = (
    "You mutate image-generation system prompts for a human-guided preference optimizer.\n"
    "Preserve the user's target aesthetic direction while addressing recurring critique patterns.\n"
    "Return only JSON with keys: compiled_prompt, rationale, preserved_traits, changed_traits, risk_notes."
)


def build_heuristic_mutation(*, parent_prompt: str | None, critiques: list[str]) -> PromptMutation:
    base_prompt = (parent_prompt or "").strip() or "Refine the image while preserving original intent."
    base = base_prompt.split("\n\nFeedback reflections:")[0].strip()
    condensed = "; ".join(c.strip() for c in critiques if c.strip())
    compiled = base if not condensed else f"{base}\n\nFeedback reflections: {condensed}"
    return PromptMutation(
        compiled_prompt=compiled,
        backend="heuristic_fallback",
        metadata={
            "rationale": "Fallback mutation appends recent feedback reflections to the parent prompt.",
            "preserved_traits": [],
            "changed_traits": [],
            "risk_notes": "",
        },
    )


def generate_prompt_mutation(
    *,
    parent_prompt: str | None,
    job_description: str,
    critiques: list[str],
    lineage_summary: dict[str, Any] | None = None,
    settings: PromptRewriteModelSettings | None = None,
    allow_env_settings: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> PromptMutation:
    active_settings = settings
    if active_settings is None:
        if not allow_env_settings:
            return build_heuristic_mutation(parent_prompt=parent_prompt, critiques=critiques)
        try:
            active_settings = PromptRewriteModelSettings.from_env()
        except ValueError:
            return build_heuristic_mutation(parent_prompt=parent_prompt, critiques=critiques)

    user_payload = {
        "job_description": job_description,
        "parent_prompt": (parent_prompt or "").strip(),
        "recent_critiques": [critique.strip() for critique in critiques if critique.strip()],
        "lineage_summary": lineage_summary or {},
    }
    payload = {
        "model": active_settings.prompt_model,
        "messages": [
            {"role": "system", "content": _MUTATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
        "temperature": 0.4,
    }
    try:
        response = requests.post(
            f"{active_settings.openrouter_base_url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {active_settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = str(response.json()["choices"][0]["message"]["content"]).strip()
        decoded = json.loads(content)
    except (requests.RequestException, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return build_heuristic_mutation(parent_prompt=parent_prompt, critiques=critiques)

    if not isinstance(decoded, dict):
        return build_heuristic_mutation(parent_prompt=parent_prompt, critiques=critiques)
    compiled_prompt = str(decoded.get("compiled_prompt") or "").strip()
    if not compiled_prompt:
        return build_heuristic_mutation(parent_prompt=parent_prompt, critiques=critiques)
    metadata = {
        "rationale": str(decoded.get("rationale") or "").strip(),
        "preserved_traits": _string_list(decoded.get("preserved_traits")),
        "changed_traits": _string_list(decoded.get("changed_traits")),
        "risk_notes": str(decoded.get("risk_notes") or "").strip(),
    }
    return PromptMutation(compiled_prompt=compiled_prompt, metadata=metadata, backend="llm_mutation")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
