# Aesthetic Job + GEPA Optimization Roadmap

## Goal

Build a job-scoped image-to-image refinement workflow:

`Select aesthetic job -> sample prompt -> extract intent -> generate baseline image -> refine image with the job policy -> require human comparison feedback -> optionally run GEPA optimization for that job`.

The image generator remains frozen. GEPA updates only the text refinement policy for the currently selected aesthetic job.

## Key Product Decisions

- Human feedback is mandatory. A rollout is not complete until the user submits an outcome and critique.
- GEPA does not run automatically after every comparison. The UI tracks eligible completed rollouts and lets the user choose when to launch optimization.
- The GEPA optimization control exposes minibatch size so the user can decide how much feedback to use per optimization pass.
- The comparison tab displays the active job's latest compiled GEPA prompt, when available.
- Different aesthetic directions are separate jobs. The online policy update applies only to the currently selected job.

## Data Model Changes

### 1. Add aesthetic jobs

Create a first-class aesthetic job concept in storage.

Suggested fields:

- `id`
- `name`
- `description`
- `status`: `active` or `archived`
- `seed_refinement_prompt`
- `active_candidate_id`
- `compiled_gepa_prompt`
- `created_at`
- `updated_at`

Purpose:

- Separates policies for different aesthetic refinements.
- Gives the UI a stable object to select before entering comparison.
- Provides the scope for rollouts, comparisons, GEPA candidates, and GEPA runs.

Files:

- Modify `src/image_preference_modelling/storage/contracts.py`
- Modify `src/image_preference_modelling/storage/state_store.py`
- Test in `tests/test_state_store.py`

Concrete steps:

1. Add `AestheticJobStatus = Literal["active", "archived"]`.
2. Add an `AestheticJobRecord` dataclass.
3. Increment `SCHEMA_VERSION`.
4. Add an `aesthetic_jobs` table.
5. Add migration logic that creates the table for existing databases.
6. Add methods:
   - `create_aesthetic_job(name, description, seed_refinement_prompt) -> str`
   - `list_aesthetic_jobs(include_archived=False) -> list[dict]`
   - `get_aesthetic_job(job_id) -> dict | None`
   - `update_aesthetic_job_policy(job_id, active_candidate_id, compiled_gepa_prompt) -> None`
7. Add storage tests for creation, listing, lookup, and policy update.

### 2. Add job-scoped rollouts

Create a rollout record for each baseline/refined pair generated in the UI.

Suggested fields:

- `id`
- `job_id`
- `prompt_text`
- `intent_text`
- `baseline_image_uri`
- `refined_image_uri`
- `candidate_id`
- `refinement_prompt`
- `model_config_json`
- `status`: `generated`, `feedback_complete`
- `created_at`
- `feedback_completed_at`

Purpose:

- Captures the full image-to-image trajectory GEPA needs.
- Makes checkpointing explicit per rollout.
- Allows GEPA runs to select only rollouts with completed human feedback.

Files:

- Modify `src/image_preference_modelling/storage/contracts.py`
- Modify `src/image_preference_modelling/storage/state_store.py`
- Test in `tests/test_state_store.py`

Concrete steps:

1. Add `RolloutStatus = Literal["generated", "feedback_complete"]`.
2. Add a `rollouts` table keyed by `job_id`.
3. Add `comparison_id` or `rollout_id` linkage so every comparison can be traced back to the generated rollout.
4. Add methods:
   - `create_rollout(...) -> str`
   - `mark_rollout_feedback_complete(rollout_id, comparison_id) -> None`
   - `list_completed_rollouts_for_job(job_id, limit=None) -> list[dict]`
5. Update integrity checks to flag rollouts whose job does not exist.
6. Add tests that verify incomplete rollouts are not returned as GEPA-eligible feedback.

### 3. Make human feedback strictly required

Require a concrete outcome and non-empty critique before a comparison can be saved.

Rules:

- `winner` outcome requires winner to be `left` or `right`.
- `both_good`, `both_bad`, and `cant_decide` are allowed, but must still include a critique.
- Empty or whitespace-only critique is rejected.
- A rollout remains `generated` until feedback is accepted.

Files:

- Modify `src/image_preference_modelling/storage/state_store.py`
- Modify `src/image_preference_modelling/gradio_app.py`
- Test in `tests/test_state_store.py`
- Test in `tests/test_gradio_app.py`

Concrete steps:

1. Add critique validation in `StateStore.add_comparison`.
2. Update UI validation so `Submit Score` shows a clear error when critique is missing.
3. Expand the UI winner/outcome control from `baseline/regenerated/tie` to include:
   - `baseline`
   - `regenerated`
   - `both_good`
   - `both_bad`
   - `cant_decide`
4. Map UI values to storage outcomes.
5. After a comparison is saved, mark the rollout as `feedback_complete`.
6. Add tests for missing critique, each valid outcome, and rollout completion.

## UI Changes

### 4. Add "Select Job -> Comparison" flow

The first UI interaction should be selecting or creating an aesthetic job.

Recommended UI shape:

1. Job selector section:
   - Dropdown of active aesthetic jobs.
   - Create job controls: name, description, seed refinement prompt.
   - "Use Selected Job" button.
2. Comparison tab:
   - Disabled until a job is selected.
   - Displays active job name.
   - Displays compiled GEPA prompt if available.
   - Runs baseline/refined comparison using the active job policy.
3. GEPA optimization controls:
   - Completed feedback count for the selected job.
   - Minibatch size input.
   - "Run GEPA Optimization" button.
   - Latest GEPA run status.

Files:

- Modify `src/image_preference_modelling/gradio_app.py`
- Test in `tests/test_gradio_app.py`

Concrete steps:

1. Add Gradio state for `active_job_id`, `active_rollout_id`, and `active_candidate_id`.
2. Add job selection controls above the comparison workflow.
3. Disable or guard comparison actions when no job is selected.
4. When generating a refinement, use:
   - `compiled_gepa_prompt` if the active job has one.
   - Otherwise `seed_refinement_prompt`.
5. Display the compiled prompt in a read-only textbox or markdown block.
6. Save a rollout after baseline and refined images exist.
7. Update `Submit Score` to require the active rollout id.
8. Refresh completed feedback count after each saved comparison.

### 5. Add explicit GEPA optimization button

GEPA optimization should be a user-triggered background run scoped to the active job.

Files:

- Modify `src/image_preference_modelling/gradio_app.py`
- Modify `src/image_preference_modelling/jobs/job_launcher.py`
- Modify `src/image_preference_modelling/storage/state_store.py`
- Test in `tests/test_job_launcher.py`
- Test in `tests/test_gradio_app.py`

Concrete steps:

1. Add a minibatch size numeric control with default `3`, minimum `1`, maximum `10`.
2. Disable or reject optimization when completed feedback count is less than minibatch size.
3. On button click, create a `gepa` run with config:
   - `job_id`
   - `minibatch_size`
   - selected completed rollout ids
   - active candidate id
   - current compiled prompt
4. Dispatch the run through `JobLauncher.dispatch_run`.
5. Show the run id and current status in the UI.
6. Add tests that the GEPA run config contains the selected job id and minibatch size.

## GEPA Candidate + Checkpoint Design

### 6. Track GEPA candidates and the Pareto frontier

Store candidate policies separately from jobs and GEPA runs.

Suggested fields for `gepa_candidates`:

- `id`
- `job_id`
- `parent_candidate_ids_json`
- `candidate_text`
- `compiled_prompt`
- `objective_scores_json`
- `frontier_member`
- `created_by_run_id`
- `created_at`

Purpose:

- Keeps all explored policies available for analysis.
- Lets each job point to its current online policy.
- Supports Pareto-frontier selection instead of a single global best prompt.

Files:

- Modify `src/image_preference_modelling/storage/contracts.py`
- Modify `src/image_preference_modelling/storage/state_store.py`
- Test in `tests/test_state_store.py`

Concrete steps:

1. Add a `gepa_candidates` table.
2. Add `create_gepa_candidate(...) -> str`.
3. Add `list_gepa_candidates_for_job(job_id) -> list[dict]`.
4. Add `set_candidate_frontier_membership(candidate_id, frontier_member) -> None`.
5. Add `promote_job_candidate(job_id, candidate_id) -> None`, which updates the job's `active_candidate_id` and `compiled_gepa_prompt`.
6. Add tests for candidate creation, listing, and promotion.

### 7. Define rollout scoring for multi-objective optimization

Each completed rollout should produce objective scores where higher is better.

Initial objectives:

- `preference_win`: regenerated beats baseline = `1.0`, baseline wins = `0.0`, non-winner outcomes = `0.5`.
- `feedback_quality`: critique is non-empty and long enough to guide reflection.
- `intent_preservation`: placeholder score of `1.0` until an evaluator exists.
- `composition_preservation`: placeholder score of `1.0` until an evaluator exists.

Later objectives:

- LLM critique alignment.
- Image similarity/composition preservation.
- Cost and latency inverse scores.
- Held-out win rate.

Files:

- Create `src/image_preference_modelling/gepa/scoring.py`
- Test in `tests/test_gepa_scoring.py`

Concrete steps:

1. Implement a pure function that maps a completed rollout + comparison to objective scores.
2. Keep placeholder image-derived objectives explicit and documented.
3. Add tests for baseline win, regenerated win, both-good, both-bad, and cannot-decide outcomes.

## Background Worker Path

### 8. Replace GEPA worker stub with job-scoped optimization behavior

The first GEPA worker can be conservative: select eligible completed rollouts, score them, write a checkpoint, and promote a simple compiled prompt. Full integration with `gepa` or `dspy.GEPA` can follow once the state boundaries are stable.

Files:

- Modify `src/image_preference_modelling/jobs/job_launcher.py`
- Create `src/image_preference_modelling/gepa/optimizer.py`
- Test in `tests/test_job_launcher.py`
- Test in `tests/test_gepa_optimizer.py`

Concrete steps:

1. Add a `GEPARunConfig` dataclass or validation helper for `gepa` run configs.
2. In `JobLauncher._execute_run`, route `run_type == "gepa"` to a GEPA worker function.
3. The worker should:
   - Load the selected job.
   - Load selected completed rollouts.
   - Score rollout objectives.
   - Build a reflection dataset from prompt, refinement prompt, winner, outcome, and critique.
   - Write `checkpoint.json` into the run artifact directory.
   - Create or update a GEPA candidate.
   - Promote the candidate to the job's online policy.
4. Keep the current simulated worker behavior for non-GEPA runs until real workers exist.
5. Add tests that a GEPA run writes `checkpoint.json` and updates the selected job policy.

## Checkpoint Contents

Each GEPA run should write a checkpoint under its run artifact directory.

Required fields:

- `run_id`
- `job_id`
- `minibatch_size`
- `selected_rollout_ids`
- `parent_candidate_id`
- `new_candidate_id`
- `compiled_prompt`
- `objective_scores`
- `frontier_snapshot`
- `created_at`

Each rollout should remain independently inspectable through storage and artifacts.

## Milestones

### M1: Job-scoped comparison loop

Deliverables:

- Aesthetic jobs exist in storage.
- UI starts with job selection.
- Comparison workflow requires an active job.
- Rollouts are saved for the active job.
- Human feedback is mandatory.

Validation:

- `uv run pytest tests/test_state_store.py tests/test_gradio_app.py -q`

### M2: Explicit GEPA run controls

Deliverables:

- UI shows completed feedback count.
- User can set minibatch size.
- User can click "Run GEPA Optimization".
- GEPA run config records job id, selected rollout ids, and minibatch size.

Validation:

- `uv run pytest tests/test_gradio_app.py tests/test_job_launcher.py -q`

### M3: Candidate policy and compiled prompt display

Deliverables:

- GEPA candidates are stored per job.
- Jobs track active candidate and compiled prompt.
- Comparison tab displays the active compiled prompt.
- New refinements use the active compiled prompt when available.

Validation:

- `uv run pytest tests/test_state_store.py tests/test_gradio_app.py -q`

### M4: First real GEPA checkpoint loop

Deliverables:

- GEPA worker writes checkpoint artifacts.
- Worker creates/promotes a candidate for the selected job.
- Objective scores are recorded.
- Pareto frontier membership is persisted.

Validation:

- `uv run pytest tests/test_gepa_scoring.py tests/test_gepa_optimizer.py tests/test_job_launcher.py -q`

### M5: GEPA package integration

Deliverables:

- Replace the conservative prompt-update worker with `gepa` or `dspy.GEPA`.
- Use completed rollouts as the feedback dataset.
- Use user critiques as natural-language reflection input.
- Keep optimization scoped to the selected aesthetic job.

Validation:

- Focused unit tests for adapter behavior.
- One online dry run only when OpenRouter credentials and image-conditioned model support are configured.

## Open Questions

- Should a new aesthetic job require a user-authored seed aesthetic brief, or should it default to the conservative refinement prompt?
- Should the comparison tab show only baseline vs one refined image, or eventually compare two Pareto candidates against each other?
- Should held-out evaluation be job-specific from the beginning, or added after the first GEPA loop is stable?

