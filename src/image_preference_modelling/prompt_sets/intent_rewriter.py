from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from image_preference_modelling.config import PromptRewriteModelSettings


class PromptRewriteClientError(RuntimeError):
    """Raised when the rewrite API call fails."""


class PromptRewriteOutputError(ValueError):
    """Raised when the rewrite model output is malformed."""


@dataclass(frozen=True)
class PromptRewriteResult:
    rewrites: dict[str, str]
    coverage: float
    used_fallback: bool


SYSTEM_INSTRUCTION = (
    "You rewrite Midjourney-style prompts into concise visual intents.\n"
    "Return JSON only with this exact schema:\n"
    '{"rewrites":[{"id":"<index_as_string>","intent":"<rewritten_prompt>"}]}\n'
    "Rules:\n"
    "- Keep the semantic intent.\n"
    "- Remove Midjourney tokens (e.g. --ar, --v, --stylize, camera metadata clutter).\n"
    "- 6-22 words per intent.\n"
    "- No markdown, prose, comments, or additional keys."
)


def _extract_response_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PromptRewriteOutputError("OpenRouter response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise PromptRewriteOutputError("OpenRouter response missing message content")
    return content.strip()


def parse_rewrite_payload(raw_content: str, expected_count: int) -> dict[int, str]:
    try:
        decoded = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise PromptRewriteOutputError(f"Rewrite response is not valid JSON: {exc}") from exc

    rewrites = decoded.get("rewrites")
    if not isinstance(rewrites, list):
        raise PromptRewriteOutputError("Rewrite response must include a rewrites list")

    parsed: dict[int, str] = {}
    for item in rewrites:
        if not isinstance(item, dict):
            raise PromptRewriteOutputError("Each rewrite item must be an object")
        idx_raw = item.get("id")
        intent = item.get("intent")
        if not isinstance(idx_raw, str) or not idx_raw.isdigit():
            raise PromptRewriteOutputError("Rewrite item id must be a digit-only string")
        if not isinstance(intent, str):
            raise PromptRewriteOutputError("Rewrite item intent must be a string")
        idx = int(idx_raw)
        intent_clean = intent.strip()
        if not intent_clean:
            continue
        if idx < 0 or idx >= expected_count:
            raise PromptRewriteOutputError(f"Rewrite id {idx} is out of range")
        parsed[idx] = intent_clean

    return parsed


class PromptIntentRewriter:
    def __init__(
        self,
        settings: PromptRewriteModelSettings,
        *,
        timeout_seconds: float = 30.0,
        coverage_threshold: float = 0.9,
    ) -> None:
        if coverage_threshold <= 0 or coverage_threshold > 1:
            raise ValueError("coverage_threshold must be in (0, 1]")
        self._settings = settings
        self._timeout_seconds = timeout_seconds
        self._coverage_threshold = coverage_threshold

    def rewrite(self, prompts: list[str]) -> PromptRewriteResult:
        if not prompts:
            return PromptRewriteResult(rewrites={}, coverage=1.0, used_fallback=False)

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompts": [{"id": str(index), "text": prompt} for index, prompt in enumerate(prompts)]
                    },
                    ensure_ascii=True,
                ),
            },
        ]

        payload = {"model": self._settings.prompt_model, "messages": messages, "temperature": 0}
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{self._settings.openrouter_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PromptRewriteClientError(f"Prompt rewrite request failed: {exc}") from exc

        content = _extract_response_content(response.json())
        parsed = parse_rewrite_payload(content, expected_count=len(prompts))

        rewrites: dict[str, str] = {}
        for index, raw_prompt in enumerate(prompts):
            rewritten = parsed.get(index)
            if rewritten:
                rewrites[raw_prompt] = rewritten

        coverage = len(rewrites) / len(prompts)
        if coverage < self._coverage_threshold:
            # Safe fallback keeps deterministic full coverage for downstream stages.
            return PromptRewriteResult(
                rewrites={prompt: prompt for prompt in prompts},
                coverage=coverage,
                used_fallback=True,
            )

        return PromptRewriteResult(rewrites=rewrites, coverage=coverage, used_fallback=False)
