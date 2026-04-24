# Generation Dry Run Config Alignment

## Summary

Align the online generation dry run with the project's actual image-generation configuration instead of using a separate dry-run settings object. The dry run should reuse the same OpenRouter configuration contract the app will use for image generation later, while keeping the Hugging Face prompt source as an internal implementation detail.

## Goals

- Reuse the existing app/OpenRouter environment contract for image generation.
- Remove the dry-run-only OpenRouter config path from `generation_pipeline.py`.
- Keep the online integration test focused on the real runtime entry point.
- Preserve the current prompt sampling behavior without expanding the public config surface.

## Non-Goals

- Making the Hugging Face prompt source user-configurable.
- Wiring the dry run into `JobLauncher` or the Gradio UI in this change.
- Unifying prompt-rewrite config and image-generation config into a single abstraction.
- Adding new test-only environment variables.

## Current State

- `config.py` contains `PromptRewriteModelSettings`, which loads `.env` and validates required environment variables for prompt rewriting.
- `generation_pipeline.py` defines `GenerationDryRunSettings`, which separately loads `.env` and reads its own environment contract for image generation.
- The dry run currently depends on dry-run-specific names such as `OPENROUTER_IMAGE_MODEL`, even though the project already exposes `IMAGE_MODEL` for image generation.
- The online integration test explicitly constructs the dry-run settings object before invoking the dry run.

## Chosen Approach

Add a shared `ImageGenerationModelSettings` type to `config.py` and make the dry run depend on that config instead of its own env-parsing dataclass.

This is the recommended middle path because it makes the dry run exercise the real config contract without broadening scope into prompt-source configurability or a larger refactor.

## Alternatives Considered

### 1. Minimal patch inside `generation_pipeline.py`

Keep the dry-run settings object but rename its environment lookups to use `IMAGE_MODEL` and the existing OpenRouter variables.

Why not chosen:
- Still duplicates config loading logic outside `config.py`.
- Leaves a dry-run-specific config type in place even though the goal is to use the actual config path.

### 2. Full shared generation config

Promote both the OpenRouter image settings and the Hugging Face prompt source into shared config.

Why not chosen:
- Solves more than the current request requires.
- Expands the public configuration surface for a prompt source that is currently just an internal dry-run input.

## Design

### Shared Config

Add `ImageGenerationModelSettings` to `config.py` with:

- `image_model`
- `openrouter_api_key`
- `openrouter_base_url`

`ImageGenerationModelSettings.from_env()` should:

- load `.env` with `load_dotenv(override=False)`,
- read `IMAGE_MODEL`,
- read `OPENROUTER_API_KEY`,
- read optional `OPENROUTER_BASE_URL` with the current default,
- raise a clear `ValueError` when required variables are missing.

The structure and error style should mirror `PromptRewriteModelSettings`.

### Generation Pipeline

`generation_pipeline.py` should stop owning the OpenRouter env contract.

Changes:

- Remove `GenerationDryRunSettings`.
- Update the OpenRouter image request code to accept `ImageGenerationModelSettings`.
- Let `run_generation_dry_run()` load `ImageGenerationModelSettings.from_env()` when a config object is not explicitly supplied.
- Keep the Hugging Face prompt source as internal defaults in the module.

The Hugging Face defaults should remain implementation details, not shared configuration:

- dataset: `daspartho/stable-diffusion-prompts`
- config: `default`
- split: `train`
- column: `prompt`

Timeout behavior should also remain internal to the pipeline rather than becoming part of the shared app config.

### Test Behavior

The online integration test should exercise the real entry point with the real config path.

Changes:

- Stop constructing or loading a dry-run-specific settings object in the test.
- Call `run_generation_dry_run(output_dir=...)` directly.
- Keep the existing assertions for:
  - prompt exists,
  - image file exists,
  - file is non-empty,
  - file is deleted successfully during cleanup.

This keeps the test aligned with the intended future runtime behavior: the caller invokes the dry-run entry point and the entry point resolves the actual config contract itself.

## Data Flow

1. The online test calls `run_generation_dry_run(output_dir=...)`.
2. `run_generation_dry_run()` loads `ImageGenerationModelSettings.from_env()`.
3. The pipeline samples one prompt from the internal Hugging Face default source.
4. The pipeline sends that prompt to OpenRouter using the shared image-generation config.
5. The returned image data URL is decoded and written to the requested output directory.
6. The test verifies file creation and then removes the file.

## Error Handling

Configuration failures should be handled in `config.py`:

- missing `IMAGE_MODEL` -> `ValueError`
- missing `OPENROUTER_API_KEY` -> `ValueError`

Operational failures should remain in `generation_pipeline.py`:

- Hugging Face request failure -> `PromptSourceClientError`
- OpenRouter request failure -> `ImageGenerationClientError`
- malformed external payloads -> `GenerationDryRunOutputError`

This keeps the split clear:

- configuration validation happens at config load time,
- remote-service and payload failures happen during execution.

## Verification Plan

- Run `uv run pytest -q`
- Run `uv run pytest --online tests/test_generation_pipeline_online.py -q`

## Expected Outcome

After this change, the dry run will no longer depend on a separate dry-run config contract for OpenRouter image generation. The online test will prove that the runtime entry point works when driven by the project's actual image-generation configuration.
