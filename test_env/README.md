---
title: ICU Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ICU Environment

An interactive reinforcement learning (RL) environment for ICU bed allocation and step‑down decisions. The environment simulates a mixed‑acuity ICU with limited beds, ventilators, and special equipment, and exposes a clean HTTP API suitable for agents running on Hugging Face Spaces or locally.

## 1. Environment Description

The ICU Environment models a hospital intensive care unit and step‑down ward. At each time step:

- New patients may arrive with varying severity (e.g. GCS score, ventilator needs, infection status, special equipment needs).
- A finite set of beds exists across multiple wards, each with capabilities (ventilator, negative pressure, special mattress, etc.).
- The agent chooses actions such as assigning a waiting patient to a compatible ICU bed or safely stepping a stable patient down out of ICU.
- The environment enforces clinical and capacity constraints and returns an updated observation, reward, and metadata.

Key API endpoints (served by `server/app.py`):

- `POST /reset` – resets the environment and returns the initial observation.
- `POST /step` – applies an action and returns the new observation and reward.
- `GET /state` – returns the current full environment state.
- `GET /health` – simple healthcheck endpoint.
- `GET /tasks` – lists available benchmark tasks.
- `POST /grader` – runs a baseline agent for a fixed number of steps.
- `GET /baseline` – runs a heuristic baseline policy until termination.

The `/web` route provides a small web dashboard that shows patients, beds, and logs, and lets you manually trigger `/reset`, `/step`, `/state`, and `/grader` from a browser.

## 2. Why This Environment?

This environment is designed to capture a realistic, high‑impact decision problem:

- **Clinically grounded constraints** – bed capabilities and patient needs must match (ventilators, infection control, special mattresses, etc.), making assignment more than a simple capacity problem.
- **Long‑horizon planning** – short‑term choices (who gets an ICU bed now, who gets stepped down) affect downstream availability, patient outcomes, and overall ICU throughput.
- **Rich observation space** – the agent observes per‑patient clinical state, ward/bed state, and high‑level hospital summaries.
- **Competition‑ready** – endpoints and schemas are standardized for easy integration into RL competitions or leaderboards.

If you want to test allocation / scheduling / triage strategies in a setting where mistakes are costly and constraints are strict, this environment provides a compact but expressive sandbox.

## 3. Setup Guide

### Local (Python) setup

1. Create and activate a virtual environment (optional but recommended):
   - `python -m venv .venv`
   - On Windows: `.venv\Scripts\activate`
2. Install dependencies from the project root (the directory containing `pyproject.toml` and `requirements.txt`):
   - `pip install -r requirements.txt`  
      or, if you use uv: `uv sync`
3. Install the environment package in editable mode (optional):
   - `pip install -e .`
4. Start the FastAPI server locally from `test_env`:
   - `cd test_env`
   - `python -m server.app`
5. The server will bind to `PORT` (default `7860`) and expose `/web`, `/docs`, `/health`, `/reset`, `/step`, `/state`, etc.

### Docker / Hugging Face Spaces

The environment ships with a Dockerfile compatible with Hugging Face Spaces (`sdk: docker`). To build and run locally:

```bash
cd test_env
docker build -t icu-env .
docker run -p 7860:7860 icu-env
```

On Spaces, push this directory as a Docker Space. The container will start the FastAPI app and listen on port `7860` as expected by the platform.

## 4. Project Structure

High‑level layout of the environment package:

- `openenv.yaml` – metadata for the OpenEnv ecosystem (name, version, description).
- `server/`
  - `app.py` – FastAPI application exposing `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`, `/web`, etc.
  - `env2.py` – implementation of the core `ICUEnvironment` dynamics (reset, step, state transitions).
- `models.py` – Pydantic/typed models for observations, actions, and internal state (patients, beds, wards, action router classes, etc.).
- `grader_v2.py` – reference grader / baseline runner using the v2 environment logic.
- `inference.py` – entry point for running agents against the environment.
- `Dockerfile` – multi‑stage build for reproducible Docker images (used by Hugging Face Spaces and local Docker runs).
- `validate-submission.sh` – helper script for validating a submission against the OpenEnv spec.

At the repo root, you also have helper scripts like `openenv_push.cmd` / `openenv_push.ps1` for pushing validated environments to OpenEnv.

## 5. Environment Variables

The server and tools understand the following environment variables:

- `PORT` – HTTP port for the FastAPI app (default: `7860`).
- `HOST` – bind address for the server (default: `0.0.0.0`).
- `.env` file – additional configuration can be injected via a `.env` file; values are loaded in `server/app.py` using `python-dotenv`.

When running on Hugging Face Spaces with `sdk: docker`, the platform sets `PORT` automatically; the Dockerfile and `main()` function use it if present.

## 6. Interaction Loop and Reward

From an RL perspective, each episode follows the standard OpenAI Gym style loop:

1. **Reset** – call `POST /reset` to obtain the initial observation `obs`.
2. **Act** – choose an action `a_t` based on the current observation.
3. **Step** – call `POST /step` with `{ "action": ... }` to advance the environment.
4. **Observe** – receive the next observation `obs'`, scalar `reward`, `done` flag, and metadata.
5. **Repeat** – continue until `done` is `true` or a step limit is reached.

On each `step` call, `ICUEnvironment.step` (see `server/env2.py`) applies the following mechanics:

- **Base time penalty** – `reward` starts at `-0.02 * len(unassigned_patients)`, encouraging the agent to clear the waiting queue quickly.
- **Health drift** – every 3 time steps, the GCS score of waiting patients is decreased by 1 (down to a lower bound), modelling deterioration while waiting.
- **ICU stay accumulation** – `days_in_icu` is incremented for all active patients, which later feeds into the step‑down efficiency bonus.
- **Decision routing** – depending on `action.action_type`, either `_assign` (for `ASSIGN_BED`) or `_step_down` (for `STEP_DOWN`) is called to compute additional dense rewards or raise clinical errors.
- **Terminal bonuses / failures** – clearing all waiting patients with no fatal clinical errors gives a success bonus and ends the episode; exceeding a step cap or accumulating too many clinical errors ends the episode as a failure.

## 7. Models Overview

Core data models are defined in `models.py` and are used both by the environment and the FastAPI schemas:

- `ICUObservation` – observation returned to the agent at each step. Includes:
  - `unassigned_patients` – list of `PatientState` objects waiting for ICU beds.
  - `active_patients` – mapping of patient IDs to `PatientState` for current ICU occupants.
  - `wards` – list of `WardState`, each containing multiple `BedState` entries.
  - `hospital_summary` – human‑readable summary string for the UI.
  - `reward`, `score`, `step`, `done`, `fatal_errors`, `feedback`.
- `PatientState` – clinical and logistical attributes for a single patient (GCS score, ventilator need, infection flag, special equipment needs, days in ICU, etc.).
- `BedState` – capacity and capabilities of a specific bed (has ventilator, negative pressure, specialized mattress, current occupant ID).
- `WardState` – grouping of beds within a ward / unit.
- `ICUActionRouter` – union‑type wrapper used by the API to route between different concrete action types.
- `AssignBedAction` – action schema for assigning a waiting patient to a bed.
- `StepDownAction` – action schema for stepping a stable ICU patient down out of ICU.

These models drive both the OpenEnv server (JSON schemas) and any client agents that want strong typing.

## 8. Task Descriptions

The `/tasks` endpoint returns metadata about the benchmark tasks included in this environment. Internally, tasks are registered in the `TASKS` dictionary in `server/env2.py`, and `ICUEnvironment` automatically cycles through them on each `reset()`.

Current built‑in tasks:

1. **`easy`**
   - **Tagged config**: see the `TASKS["easy"]` block in `server/env2.py`.
   - **Description**: "Direct triage. Beds are plentiful. Assign based on basic constraints." Two patients arrive: one ventilated, one infectious. There are ample ICU beds with ventilators and isolation, so the focus is on correctly matching patient needs to appropriate beds without violating constraints.
   - **Goal for the agent**: Safely admit all waiting patients by assigning each to a clinically compatible bed, with minimal delay and no clinical errors.

2. **`medium`**
   - **Tagged config**: see the `TASKS["medium"]` block in `server/env2.py`.
   - **Description**: "Constraint heavy. Patients have competing specialty needs." The patient mix includes a paralysis case needing a specialty mattress and a severe head‑injury patient who is ventilated. Capacity is more limited, and different wards expose different equipment (ventilators, negative pressure, specialty mattresses).
   - **Goal for the agent**: Correctly route each patient to a bed that satisfies all specialty requirements while respecting infection control and ventilator constraints. The agent must reason about multiple overlapping clinical needs.

3. **`hard`**
   - **Tagged config**: see the `TASKS["hard"]` block in `server/env2.py`, including the pre‑admitted "dummy" occupant.
   - **Description**: "Capacity crisis. The agent must step-down a stable patient to free a bed." A high‑acuity ventilated patient arrives when a key ICU bed (e.g. `S1`) is already occupied by a very stable patient with high GCS and several days in ICU.
   - **Goal for the agent**: Recognize that the stable occupant is safe to step down (using the `STEP_DOWN` action) in order to free a ventilator‑capable bed for the new critical arrival. This scenario explicitly tests the agent’s ability to trade off continued ICU occupancy against incoming demand.

You can add more tasks by extending the `TASKS` registry in `server/env2.py` and, if desired, surfacing them through the `/tasks` endpoint in `server/app.py` (e.g. with task IDs, descriptions, and difficulty tags).

## 9. Evaluation System

The environment provides built‑in evaluation helpers so you can easily benchmark agents:

- **Grader endpoint** – `POST /grader?steps=N`
  - Runs a simple random baseline policy for `N` steps on the v2 environment.
  - Returns aggregate metrics such as final `score`, `total_reward`, `steps`, `fatal_errors`, and `feedback`.
- **Baseline endpoint** – `GET /baseline`
  - Runs a heuristic, clinically informed baseline until the episode terminates.
  - Tries to assign patients to the first feasible bed satisfying all clinical constraints, then safely steps down stable ICU patients when appropriate.

These tools are intended both as sanity checks (to ensure the environment is wired correctly) and as reference points for more advanced RL agents.

## 10. Reward Function

The scalar `reward` returned in each `ICUObservation` is built up in `ICUEnvironment.step` using helper decision functions `_assign` and `_step_down` (see `server/env2.py`). In addition to the base time penalty, the main components are:

- **Assignment reward (`_assign`)**
  - After passing all clinical constraint checks (ventilator, isolation, specialty mattress), the environment executes the bed assignment and adds:
    - `base_reward = 0.2` – credit for any successful admission.
    - `urgency_bonus = (15 - gcs_score) * 0.02` – higher bonus for more critical patients (lower GCS), incentivizing prioritization of sicker patients.
    - `vent_bonus = 0.1` if `needs_ventilator` else `0.0` – extra credit for correctly placing ventilated patients.
  - The total contribution from `_assign` is `base_reward + urgency_bonus + vent_bonus`.

- **Step‑down reward (`_step_down`)**
  - Only allowed if the patient is clinically stable: GCS ≥ 14 and not ventilator‑dependent. Violations raise clinical error exceptions and incur a penalty in `step`.
  - After a valid step‑down, the environment frees the corresponding bed and removes the patient from ICU, then adds:
    - `base_reward = 0.2` – credit for any safe discharge from ICU.
    - `days_past_recovery = max(0, days_in_icu - 3)` – days beyond a nominal recovery threshold.
    - `efficiency_bonus = days_past_recovery * 0.05` – rewarding timely step‑downs after patients have been in ICU for several days.
  - The total contribution from `_step_down` is `base_reward + efficiency_bonus`.

- **Clinical error penalty**
  - If `_assign` or `_step_down` raises an exception (e.g. wrong bed type, unsafe step‑down), `step` subtracts `0.5` from the reward, increments `fatal_errors`, and returns feedback explaining the error. The episode is not immediately terminated, allowing agents to recover, but too many such errors will end the episode as a failure.

- **Episode‑level bonuses / failures**
  - Clearing all `unassigned_patients` with `fatal_errors == 0` grants an additional `+1.0` reward and ends the episode with the message "SUCCESS: Shift completed safely."
  - If the agent exceeds `max_steps` or accumulates `fatal_errors >= 3`, the episode ends with "FAILED: Shift ended or too many clinical errors.", and the final score is clamped into the OpenEnv range using `_clamp`.

You can tune the behavior of the environment by editing `ICUEnvironment.step`, `_assign`, and `_step_down` in `server/env2.py` (for example, changing the base rewards, urgency scaling, or clinical error penalties).
