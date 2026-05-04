# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Local research tool for optimizing image-generation prompts against personal preferences. The pipeline runs: sample a prompt → generate a baseline image → generate a candidate image using an evolving system prompt → collect pairwise human ratings with critiques → generate prompt mutations from accumulated feedback → evaluate and promote the best candidates.

A Gradio web UI ("Gradio Operator Cockpit") is the control plane for all pipeline stages.

## Commands

Requires Python 3.13+ (see `pyproject.toml`); `.python-version` pins the interpreter for `uv`.

```bash
# Install dependencies (uses uv)
uv sync

# Run the Gradio UI
uv run gradio-cockpit

# Run all tests (skips tests that call real external services)
uv run pytest

# Run tests that call real external services (OpenRouter, HuggingFace)
uv run pytest --online

# Run a single test file
uv run pytest tests/test_state_store.py

# Run a single test by name
uv run pytest tests/test_job_launcher.py::test_cancel_running_run_transitions_to_cancelled
```

Tests that hit external services must be decorated with `@pytest.mark.online`. The marker is registered in `tests/conftest.py` and is skipped unless `--online` is passed.

## Environment variables

Loaded from `.env` via `python-dotenv` (does not override already-exported vars):

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Auth key for OpenRouter (required by all model calls) |
| `IMAGE_MODEL` | OpenRouter model ID for image generation (required by `ImageGenerationModelSettings`) |
| `PROMPT_MODEL` | OpenRouter model ID for prompt rewriting, critique judging, mutation generation, and LLM-guided prompt selection |
| `OPENROUTER_BASE_URL` | Rarely set; defaults to `https://openrouter.ai/api/v1` |

Both `PromptRewriteModelSettings.from_env()` and `ImageGenerationModelSettings.from_env()` raise `ValueError` listing missing vars if their required keys are absent.

## Architecture

The package lives in `src/image_preference_modelling/` with these layers:

### Config (`config.py`)
Two frozen dataclasses loaded from environment:
- `PromptRewriteModelSettings` — for intent rewriting, LLM-guided prompt selection, critique judging, and prompt mutation (`PROMPT_MODEL`)
- `ImageGenerationModelSettings` — for image generation calls (`IMAGE_MODEL`)

### Prompt Sets (`prompt_sets/`)
`intent_rewriter.py` — `PromptIntentRewriter` strips Midjourney tokens (e.g. `--ar`, `--v`, `--stylize`) and rewrites raw prompts into concise visual intents via an OpenRouter chat completion call. Returns strict JSON (`{"rewrites":[{"id":"<index>","intent":"..."}]}`). If coverage falls below `coverage_threshold`, falls back to identity mapping and sets `used_fallback=True`.

### Generation Pipeline (`generation_pipeline.py`)
Handles two concerns:

1. **Prompt sampling** — downloads the `succinctly/midjourney-prompts` HuggingFace dataset as a parquet file into `data/prompt_sources/` on first use, then samples from it. Sampling is either random or LLM-guided: given a job description and category, it sends a batch of candidate prompts to the LLM and asks for match/score assessments (`{"assessments":[{"id":...,"match":bool,"score":0-1,"reason":"..."}]}`), then selects the best match.

2. **Image generation** — calls OpenRouter's `/chat/completions` endpoint with `"modalities": ["image"]` and decodes the base64 data URL response. `generate_image_from_openrouter` generates a baseline (no system prompt); `generate_candidate_image_from_openrouter` applies the job's current system prompt. Images are saved to `data/jobs/<job_id>/images/<rollout_id>/`.

### Storage (`storage/`)
- `contracts.py` — type aliases (`RunType`, `RunStatus`, `RatingOutcome`, `AestheticJobStatus`, `RolloutStatus`) and artifact directory helpers.
- `state_store.py` — `StateStore` wraps SQLite at `.local/state/cockpit.db` (schema versioned, currently v11). Migrations run on `__init__`; `_migrate_vN_to_vN+1` methods are append-only — never edit a past migration. Key entities:
  - **aesthetic_jobs** — named optimization targets, each with a `seed_system_prompt`, `seed_candidate_id`, `latest_system_prompt`, `baseline_system_prompt`, `active_candidate_id`, and a mutation threshold. Creating or migrating a job inserts the seed prompt into `gepa_candidates` as an evaluated frontier seed candidate.
  - **rollouts** — generalized to three types (`VALID_ROLLOUT_TYPES`):
    - `baseline_candidate` — baseline image (no system prompt) vs candidate image (job's active prompt). Left = baseline, right = candidate.
    - `candidate_comparison` — two saved candidates compared head-to-head; both `left_candidate_id` and `right_candidate_id` are set, with separate `left_system_prompt_snapshot` and `right_system_prompt_snapshot`.
    - `latest_prompt_check` — used by the Best Candidate Check tab to sanity-check the current best training candidate against a no-system baseline.
    - Rollouts also record generation mode (`image_conditioned` or `text_only`). Status is `generated` until feedback is submitted, then `feedback_complete`.
  - **gepa_candidates** — seed prompts and system prompt versions produced by prompt-mutation runs. Candidates have statuses `proposed`, `evaluating`, `evaluated`, or `archived`; track `elo`, `score`, `confidence`, win/loss/tie counts, judge metadata, parent IDs, and Pareto frontier membership.
  - **runs / run_events** — job execution records; artifact dirs use `RUN_TYPE_ARTIFACT_DIR` (`generation_runs`, `reward_model_versions`, `gepa_runs`, `evaluation_runs`) under `.local/artifacts/`, with `config.json`, run events, `job.log`, and for GEPA runs a `checkpoint.json`.
  - **rating_sessions / comparisons** — raw pairwise feedback store.

  Run status transitions are strictly enforced: `queued → running | cancelled`, `running → completed | failed | cancelled`. Terminal states have no outgoing transitions.

### Prompt Evolution (`gepa/`)
- `scoring.py` — `score_rollout_feedback` converts completed rollout feedback into initial objective signals.
- `reward.py` — pure Elo, confidence, and blended-score helpers for human head-to-head candidate evaluation.
- `critique_judge.py` — asks `PROMPT_MODEL` to interpret the human critique and estimate preference margin / critique confidence. The human winner remains authoritative; the LLM never flips the outcome.
- `mutation_engine.py` — generates prompt mutations from a parent prompt, job description, lineage summary, and recent critiques. If model configuration is unavailable, falls back to appending feedback reflections.
- `optimizer.py` — `run_gepa_optimization` is now the prompt-mutation worker kept under the existing `gepa` run type for compatibility. It loads selected feedback, picks a parent from the evaluated frontier when available, creates one proposed candidate, and writes a checkpoint. Promotion requires later human evaluation.

  Mutation gating: `StateStore.get_gepa_gate_status` enables mutation once `new_feedback_count >= threshold` (default 2), where `new_feedback_count` is the count of `feedback_complete` rollouts whose `feedback_completed_at` is later than the last completed mutation run's `finished_at` for that job. The UI can auto-start a mutation when no proposed/evaluating candidate or active mutation run is waiting.

  Important semantics:
  - The UI winner control is intentionally three-way: `left`, `right`, or `no_clear_winner`. Legacy `both_good`, `both_bad`, and `cant_decide` map to `no_clear_winner`.
  - Training matchup selection prioritizes pending proposed/evaluating candidates against frontier/evaluated candidates. If no pending candidates exist, evaluated non-seed candidates are compared against the seed candidate before falling back to active/latest-vs-baseline.
  - Human feedback controls credit direction. The critique judge only estimates update strength (`winner_margin`) and critique usefulness (`critique_confidence`).
  - Candidate rewards are stored as `elo`, `score`, and `confidence`. `score` is a blended ranking signal; promotion still requires an evaluated frontier candidate.
  - Archived candidates stay in the prompt pool for inspection but are excluded from pending/evaluated selection queries. The Prompt Pool Explorer can archive or unarchive a selected candidate; active promoted candidates cannot be archived, and unarchiving restores candidates with evidence to `evaluated` and others to `proposed`.
  - Do not add `dspy` back for this loop. The project is intentionally using a handrolled, human-guided mutation/evaluation loop because the reward is subjective image preference.

### Jobs (`jobs/`)
`job_launcher.py` — `JobLauncher` dispatches runs onto a `ThreadPoolExecutor` (max 2 workers). `RunType` is `Literal["generation", "reward_model", "gepa", "evaluation"]`, but only `gepa` is wired to real work (`run_gepa_optimization`); the other three currently route to a simulation stub controlled by `simulated_steps` / `simulated_step_seconds` config keys, or `force_fail: true`. Cancellation is cooperative: `cancel_run` sets a flag that the worker checks.

### UI (`gradio_app.py`, `app_context.py`)
`AppContext` is a frozen dataclass holding a `StateStore` and `JobLauncher`. `build_app(context)` accepts an injected context (useful for testing); `default_context()` builds from `.local/` in the working directory.

The cockpit is split into workflow tabs: job setup, a training comparison loop (sample prompt → prepare matchup → generate left/right images → submit winner/no-clear-winner + critique), Prompt Evolution, Best Candidate Check, and Rollout Inspector. Prompt Evolution includes mutation controls/logs plus a Prompt Pool Explorer for inspecting all `gepa_candidates`, viewing compiled prompts/metadata, and archiving or unarchiving selected prompts. Feedback updates candidate Elo/score/confidence; enough new feedback can auto-start a prompt mutation, while promotion remains explicit.

Best Candidate Check selects `StateStore.get_best_training_candidate(...)`: highest `score`, then `elo`, then `confidence`, then recency among evaluated candidates with enough evidence. It compares that candidate against a no-system baseline. It does not necessarily use `job.active_candidate_id` or `job.latest_system_prompt`; those represent explicitly promoted production state.

### Local data
All runtime state lives under `.local/` (gitignored). Downloaded prompt parquets live in `data/prompt_sources/` and job images in `data/jobs/`. Tests use pytest's `tmp_path` fixture and never touch `.local/`.
