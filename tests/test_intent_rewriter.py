from __future__ import annotations

import pytest

from image_preference_modelling.config import PromptRewriteModelSettings
from image_preference_modelling.prompt_sets.intent_rewriter import (
    PromptIntentRewriter,
    PromptRewriteOutputError,
    parse_rewrite_payload,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return self._payload


def test_parse_rewrite_payload_accepts_strict_schema() -> None:
    parsed = parse_rewrite_payload(
        raw_content='{"rewrites":[{"id":"0","intent":"red fox in snowy forest"},{"id":"1","intent":"vintage sci-fi city skyline"}]}',
        expected_count=2,
    )
    assert parsed == {0: "red fox in snowy forest", 1: "vintage sci-fi city skyline"}


def test_parse_rewrite_payload_rejects_non_json() -> None:
    with pytest.raises(PromptRewriteOutputError, match="not valid JSON"):
        parse_rewrite_payload(raw_content="not-json", expected_count=1)


def test_parse_rewrite_payload_rejects_out_of_range_ids() -> None:
    with pytest.raises(PromptRewriteOutputError, match="out of range"):
        parse_rewrite_payload(
            raw_content='{"rewrites":[{"id":"3","intent":"test"}]}',
            expected_count=2,
        )


def test_rewrite_falls_back_when_coverage_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = ["fox --ar 3:2 --v 6", "retro robot portrait --stylize 700"]
    settings = PromptRewriteModelSettings(prompt_model="test-model", openrouter_api_key="secret")

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"rewrites":[{"id":"0","intent":"fox in natural habitat"}]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("image_preference_modelling.prompt_sets.intent_rewriter.requests.post", _fake_post)

    rewriter = PromptIntentRewriter(settings, coverage_threshold=0.8)
    result = rewriter.rewrite(prompts)

    assert result.coverage == 0.5
    assert result.used_fallback is True
    assert result.rewrites == {prompt: prompt for prompt in prompts}


def test_rewrite_preserves_raw_to_rewritten_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = ["night cityscape --ar 16:9", "minimal logo mark --v 5.2"]
    settings = PromptRewriteModelSettings(prompt_model="test-model", openrouter_api_key="secret")

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"rewrites":['
                                '{"id":"0","intent":"night city skyline with reflections"},'
                                '{"id":"1","intent":"minimal geometric logo symbol"}'
                                "]}"
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("image_preference_modelling.prompt_sets.intent_rewriter.requests.post", _fake_post)

    rewriter = PromptIntentRewriter(settings, coverage_threshold=1.0)
    result = rewriter.rewrite(prompts)

    assert result.used_fallback is False
    assert result.coverage == 1.0
    assert result.rewrites == {
        "night cityscape --ar 16:9": "night city skyline with reflections",
        "minimal logo mark --v 5.2": "minimal geometric logo symbol",
    }

