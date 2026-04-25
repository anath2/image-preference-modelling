from __future__ import annotations

import pytest

import image_preference_modelling.config as config_module
from image_preference_modelling.config import ImageGenerationModelSettings


@pytest.fixture(autouse=True)
def stub_load_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dotenv(*args: object, **kwargs: object) -> None:
        assert args == ()
        assert kwargs == {"override": False}

    monkeypatch.setattr(config_module, "load_dotenv", fake_load_dotenv)


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
