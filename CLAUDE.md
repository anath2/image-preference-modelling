# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Local research tool for optimizing image-generation prompts against personal preferences. The pipeline runs: generate images from prompts → collect pairwise human ratings → train a reward model → run a GEPA (genetic/evolutionary prompt rewriter) → evaluate.

A Gradio web UI ("Gradio Operator Cockpit") is the control plane for all pipeline stages.

## Commands

```bash
# Install dependencies (uses uv)
uv sync

# Run the Gradio UI
uv run gradio-cockpit

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_state_store.py

# Run a single test by name
uv run pytest tests/test_job_launcher.py::test_cancel_running_run_transitions_to_cancelled
```

## Environment variables

Required for prompt rewriting (loaded from `.env` via `python-dotenv`, does not override exported vars):

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Auth key for OpenRouter |
| `PROMPT_MODEL` | Model ID to use for intent rewriting (e.g. an OpenAI-compatible model via OpenRouter) |
| `OPENROUTER_BASE_URL` | Optional override; defaults to `https://openrouter.ai/api/v1` |

`PromptRewriteModelSettings.from_env()` raises `ValueError` with a clear list of what's missing if required vars are absent.

## Architecture

The package lives in `src/image_preference_modelling/` with four layers:

### Config (`config.py`)
`PromptRewriteModelSettings` — frozen dataclass loaded from environment. Single source of truth for LLM API settings; constructed once and injected into `PromptIntentRewriter`.

### Prompt Sets (`prompt_sets/`)
`intent_rewriter.py` — `PromptIntentRewriter` strips Midjourney tokens (e.g. `--ar`, `--v`, `--stylize`) and rewrites raw prompts into concise visual intents via an OpenRouter chat completion call. Input/output contract: raw prompt string → rewritten intent string, keyed by the original prompt text. The model is instructed to return strict JSON (`{"rewrites":[{"id":"<index>","intent":"..."}]}`); `parse_rewrite_payload` validates it. If coverage (fraction of prompts successfully rewritten) falls below `coverage_threshold`, the rewriter falls back to identity mapping and sets `used_fallback=True` so downstream stages can detect degraded quality.

### Storage (`storage/`)
- `contracts.py` — type aliases (`RunType`, `RunStatus`, `RatingOutcome`) and `RunRecord` dataclass. `artifact_path()` is the canonical helper for locating run artifacts.
- `state_store.py` — `StateStore` wraps SQLite at `.local/state/cockpit.db`. Four tables: `runs`, `rating_sessions`, `comparisons`, `run_events`. Every run also gets an artifact directory at `.local/artifacts/<run_id>/` containing `config.json` and a `job.log` written on completion.

Run status transitions are strictly enforced: `queued → running | cancelled`, `running → completed | failed | cancelled`. Terminal states have no outgoing transitions.

### Jobs (`jobs/`)
`job_launcher.py` — `JobLauncher` dispatches runs onto a `ThreadPoolExecutor` (max 2 workers). The current `_execute_run` is a stub simulating work via `simulated_steps` / `simulated_step_seconds` config keys, or raising on `force_fail: true`. Real domain workers will replace this stub. Cancellation is cooperative: `cancel_run` sets a flag that the worker checks between steps.

### UI (`gradio_app.py`, `app_context.py`)
`AppContext` is a frozen dataclass holding a `StateStore` and `JobLauncher`. `build_app(context)` accepts an injected context (useful for testing); `default_context()` builds from `.local/` in the working directory. Tabs: Overview metrics, Prompt Sets, Runs (create/dispatch/cancel/log), Review Queue (pairwise comparisons), Reward Model, Rewriter/GEPA, Evaluation.

### Local data
All runtime state lives under `.local/` (gitignored). Tests use pytest's `tmp_path` fixture and never touch `.local/`.
