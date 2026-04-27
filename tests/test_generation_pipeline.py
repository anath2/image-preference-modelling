from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest

from image_preference_modelling.config import ImageGenerationModelSettings
from image_preference_modelling import generation_pipeline
from image_preference_modelling.generation_pipeline import (
    GenerationDryRunOutputError,
    run_generation_dry_run,
)


class _FakeResponse:
    def __init__(self, *, json_payload: object | None = None, content: bytes = b"") -> None:
        self._json_payload = json_payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._json_payload


def _image_payload(data_url: str = "data:image/png;base64,aGVsbG8=") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "images": [
                        {
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    ]
                }
            }
        ]
    }


def _capture_openrouter_post(
    captured_payload: dict[str, object],
    *,
    expected_url: str,
    expected_auth_header: str,
    expected_timeout: float,
) -> callable:
    def _fake_post(
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        assert url == expected_url
        assert headers["Authorization"] == expected_auth_header
        assert timeout == expected_timeout
        captured_payload.update(json)
        return _FakeResponse(json_payload=_image_payload())

    return _fake_post


def _prompt_parquet_bytes(prompts: list[str]) -> bytes:
    pa = importlib.import_module("pyarrow")
    pq = importlib.import_module("pyarrow.parquet")
    buffer = io.BytesIO()
    pq.write_table(pa.table({"text": prompts}), buffer)
    return buffer.getvalue()


def _cached_prompt_path(prompt_source_root: Path) -> Path:
    return (
        prompt_source_root
        / "succinctly_midjourney-prompts"
        / "default"
        / "train.parquet"
    )


def test_ensure_prompt_source_parquet_persists_train_split_under_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []
    parquet_bytes = _prompt_parquet_bytes(["A detailed watercolor castle under violet skies"])

    def _fake_get(
        url: str,
        *,
        params: dict[str, str] | None = None,
        timeout: float,
    ) -> _FakeResponse:
        calls.append((url, params))
        if url == "https://datasets-server.huggingface.co/parquet":
            assert params == {"dataset": "succinctly/midjourney-prompts"}
            return _FakeResponse(
                json_payload={
                    "parquet_files": [
                        {
                            "config": "default",
                            "split": "train",
                            "url": "https://example.test/train.parquet",
                        }
                    ]
                }
            )
        assert url == "https://example.test/train.parquet"
        assert params is None
        return _FakeResponse(content=parquet_bytes)

    monkeypatch.setattr(generation_pipeline.requests, "get", _fake_get)

    parquet_path = generation_pipeline.ensure_prompt_source_parquet(
        prompt_source_root=tmp_path / "data" / "prompt_sources"
    )

    assert parquet_path == (
        tmp_path
        / "data"
        / "prompt_sources"
        / "succinctly_midjourney-prompts"
        / "default"
        / "train.parquet"
    )
    assert parquet_path.read_bytes() == parquet_bytes
    assert calls == [
        ("https://datasets-server.huggingface.co/parquet", {"dataset": "succinctly/midjourney-prompts"}),
        ("https://example.test/train.parquet", None),
    ]


def test_ensure_prompt_source_parquet_reuses_valid_cached_file_without_refetching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_source_root = tmp_path / "data" / "prompt_sources"
    parquet_path = _cached_prompt_path(prompt_source_root)
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(_prompt_parquet_bytes(["A detailed watercolor castle under violet skies"]))

    def _fail_if_refetched(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached parquet should not be refetched")

    monkeypatch.setattr(generation_pipeline.requests, "get", _fail_if_refetched)

    assert generation_pipeline.ensure_prompt_source_parquet(prompt_source_root=prompt_source_root) == parquet_path


def test_sample_prompts_from_local_source_filters_and_samples_local_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "data" / "prompt_sources" / "cached.parquet"
    prompts = [
        "16:9",
        "--uplight",
        "cat",
        "A cinematic portrait of a glass astronaut walking through neon rain",
        "Wide angle photo of an ancient library with floating candles and blue fog",
        "An editorial fashion photo of a bronze robot beside orchids",
        "A panoramic fantasy harbor at sunrise with pearl airships arriving",
    ]
    sampled_population: list[str] = []

    monkeypatch.setattr(
        generation_pipeline,
        "ensure_prompt_source_parquet",
        lambda **kwargs: parquet_path,
    )
    monkeypatch.setattr(
        generation_pipeline,
        "read_prompts_from_parquet",
        lambda path, *, prompt_column: prompts,
    )
    monkeypatch.setattr(
        generation_pipeline.random,
        "sample",
        lambda population, *, k: sampled_population.extend(population) or population[:k],
    )

    sampled_prompts = generation_pipeline.sample_prompts_from_local_source(
        prompt_source_root=tmp_path / "data" / "prompt_sources",
        candidate_count=2,
    )

    assert sampled_prompts == [
        "A cinematic portrait of a glass astronaut walking through neon rain",
        "Wide angle photo of an ancient library with floating candles and blue fog",
    ]
    assert sampled_population == [
        "A cinematic portrait of a glass astronaut walking through neon rain",
        "Wide angle photo of an ancient library with floating candles and blue fog",
        "An editorial fashion photo of a bronze robot beside orchids",
        "A panoramic fantasy harbor at sunrise with pearl airships arriving",
    ]


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
        "image_preference_modelling.generation_pipeline.sample_prompts_from_local_source",
        lambda prompt_source_root=Path("data/prompt_sources"), candidate_count=20, timeout_seconds=60.0: [
            "test prompt"
        ],
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

    def _fake_refine_image(
        prompt: str,
        source_image_data_url: str,
        settings: ImageGenerationModelSettings,
        *,
        timeout_seconds: float = 60.0,
    ) -> str:
        calls["refine_prompt"] = prompt
        calls["source_image_data_url"] = source_image_data_url
        calls["refine_settings"] = settings
        calls["refine_timeout_seconds"] = timeout_seconds
        return "data:image/png;base64,d29ybGQ="

    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.generate_image_from_openrouter",
        _fake_generate_image,
    )
    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.generate_image_refinement_from_openrouter",
        _fake_refine_image,
    )

    result = run_generation_dry_run(output_dir=tmp_path / "data")

    assert calls["prompt"] == "test prompt"
    assert calls["settings"] == resolved_settings
    assert calls["timeout_seconds"] == 60.0
    assert calls["refine_settings"] == resolved_settings
    assert calls["refine_timeout_seconds"] == 60.0
    assert isinstance(calls["source_image_data_url"], str)
    assert str(calls["source_image_data_url"]).startswith("data:image/png;base64,")
    assert result.prompt == "test prompt"
    assert result.image_path.exists()
    assert result.baseline_image_path.exists()
    assert result.refined_image_path.exists()
    assert result.image_path == result.refined_image_path
    assert result.image_path.parent == tmp_path / "data"


def test_run_generation_dry_run_tries_next_candidate_after_payload_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = ImageGenerationModelSettings(
        image_model="openai/gpt-5.4-image-2",
        openrouter_api_key="secret",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    attempted_prompts: list[str] = []

    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.sample_prompts_from_local_source",
        lambda prompt_source_root=Path("data/prompt_sources"), candidate_count=20, timeout_seconds=60.0: [
            "unsafe candidate",
            "successful candidate",
        ],
    )

    def _fake_generate_image(
        prompt: str,
        settings: ImageGenerationModelSettings,
        *,
        timeout_seconds: float = 60.0,
    ) -> str:
        attempted_prompts.append(prompt)
        if prompt == "unsafe candidate":
            raise GenerationDryRunOutputError("OpenRouter response missing images")
        return "data:image/png;base64,aGVsbG8="

    def _fake_refine_image(
        prompt: str,
        source_image_data_url: str,
        settings: ImageGenerationModelSettings,
        *,
        timeout_seconds: float = 60.0,
    ) -> str:
        return "data:image/png;base64,d29ybGQ="

    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.generate_image_from_openrouter",
        _fake_generate_image,
    )
    monkeypatch.setattr(
        "image_preference_modelling.generation_pipeline.generate_image_refinement_from_openrouter",
        _fake_refine_image,
    )

    result = run_generation_dry_run(output_dir=tmp_path / "data", settings=settings)

    assert attempted_prompts == ["unsafe candidate", "successful candidate"]
    assert result.prompt == "successful candidate"
    assert result.image_path.exists()
    assert result.baseline_image_path.exists()
    assert result.refined_image_path.exists()


def test_generate_image_from_openrouter_requests_image_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ImageGenerationModelSettings(
        image_model="bytedance-seed/seedream-4.5",
        openrouter_api_key="secret",
        openrouter_base_url="https://openrouter.example/api/v1",
    )
    captured_payload: dict[str, object] = {}

    monkeypatch.setattr(
        generation_pipeline.requests,
        "post",
        _capture_openrouter_post(
            captured_payload,
            expected_url="https://openrouter.example/api/v1/chat/completions",
            expected_auth_header="Bearer secret",
            expected_timeout=60.0,
        ),
    )

    image_data_url = generation_pipeline.generate_image_from_openrouter(
        "A cinematic fox in a moonlit forest",
        settings,
    )

    assert captured_payload["model"] == "bytedance-seed/seedream-4.5"
    assert captured_payload["modalities"] == ["image"]
    assert image_data_url == "data:image/png;base64,aGVsbG8="


def test_generate_image_refinement_from_openrouter_sends_text_and_image_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ImageGenerationModelSettings(
        image_model="google/gemini-2.5-flash-image",
        openrouter_api_key="secret",
        openrouter_base_url="https://openrouter.example/api/v1",
    )
    captured_payload: dict[str, object] = {}

    monkeypatch.setattr(
        generation_pipeline.requests,
        "post",
        _capture_openrouter_post(
            captured_payload,
            expected_url="https://openrouter.example/api/v1/chat/completions",
            expected_auth_header="Bearer secret",
            expected_timeout=60.0,
        ),
    )

    image_data_url = generation_pipeline.generate_image_refinement_from_openrouter(
        "Refine this image",
        "data:image/png;base64,dGVzdA==",
        settings,
    )

    assert captured_payload["model"] == "google/gemini-2.5-flash-image"
    assert captured_payload["modalities"] == ["image"]
    messages = captured_payload["messages"]
    assert isinstance(messages, list)
    first_message = messages[0]
    assert isinstance(first_message, dict)
    content = first_message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Refine this image"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,dGVzdA=="},
    }
    assert image_data_url == "data:image/png;base64,aGVsbG8="
