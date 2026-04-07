envs/my_env/

# ICU Bed Allocation RL Environment

This repository implements a small ICU bed allocation environment on top of OpenEnv.
It includes:

- A structured environment with ICU-specific patient and bed logic.
- An HTTP server exposing the environment via FastAPI.
- Stand‑alone graders / baselines for quick evaluation.

The new environment and server you will most often use are:

- Environment logic: [test_env/server/env2.py](test_env/server/env2.py)
- Pydantic models: [test_env/models2.py](test_env/models2.py)
- FastAPI app (v2): [test_env/server/app2.py](test_env/server/app2.py)
- Python grader v2: [test_env/grader_v2.py](test_env/grader_v2.py)

---

## Environment v2: ICUEnvironment

The v2 environment in [test_env/server/env2.py](test_env/server/env2.py) exposes ICU‑style triage dynamics.

### Core methods

- `reset()`
  - Resets episode time, reward and error counters.
  - Seeds a small waiting queue of patients (`PT-01` .. `PT-03`).
  - Seeds one stable ICU patient (`PT-04`) already occupying bed `S4`.
  - Defines one ward (`Standard ICU`) with four beds (`S1`..`S4`) with different capabilities.
  - Returns an `ICUObservation` (see below).

- `step(action: ICUAction)`
  - Increments `time`.
  - Applies a small queue penalty: `-0.02 * len(unassigned_patients)`.
  - Every 3 steps, waiting patients may deteriorate (GCS decreases by 1 down to 3).
  - Active ICU patients accrue `days_in_icu`.
  - Supports two action types:
    - `ASSIGN_BED` → handled by `_assign(AssignBedAction)`.
    - `STEP_DOWN` → handled by `_step_down(StepDownAction)`.
  - If a clinical constraint is violated, raises an exception which is converted into:
    - `reward -= 0.5`, `fatal_errors += 1`, `done = True`, and an error `feedback`.
  - If all patients are assigned and `fatal_errors == 0`, adds a +1.0 success bonus and ends the episode.
  - If `time >= max_steps` (default 20), ends the episode as a failure.
  - Returns a new `ICUObservation`.

- `state` (property)
  - Returns a Python `dict` with the underlying simulator state:
    - `"unassigned_patients"`: `List[PatientState]`
    - `"active_patients"`: `Dict[str, PatientState]`
    - `"wards"`: `List[WardState]`

### Models

All models used by v2 live in [test_env/models2.py](test_env/models2.py).

- `PatientState`
  - `patient_id: str`
  - `age: int`
  - `gcs_score: int`
  - `needs_ventilator: bool`
  - `is_infectious: bool`
  - `has_severe_head_injury: bool`
  - `has_paralysis: bool`
  - `days_in_icu: int`

- `BedState`
  - `bed_id: str`
  - `has_ventilator: bool`
  - `is_negative_pressure: bool`
  - `has_specialized_mattress: bool`
  - `current_occupant_id: Optional[str]`

- `WardState`
  - `ward_name: str`
  - `beds: List[BedState]`

- Actions (`ICUAction` is a union of):
  - `AssignBedAction`
    - `action_type = "ASSIGN_BED"`
    - `patient_id: str`
    - `bed_id: str`
  - `StepDownAction`
    - `action_type = "STEP_DOWN"`
    - `patient_id: str`

- `ICUObservation`
  - `hospital_summary: str` – human‑friendly text listing empty beds.
  - `unassigned_patients: List[PatientState]` – current waiting queue.
  - `feedback: str` – textual feedback for the last action.
  - `reward: float` – scalar reward for the last transition.
  - `done: bool` – episode termination flag.
  - `metadata: dict` – extra signals, including:
    - `"score"`: bounded episode‑level score in `[0, 1]`.
    - `"step"`: current time step.
    - `"fatal_errors"`: number of fatal clinical errors.

---

## FastAPI Server (app2.py)

The v2 HTTP API is implemented in [test_env/server/app2.py](test_env/server/app2.py).

It uses `create_fastapi_app(ICUEnvironment, ICUAction, ICUObservation)` to expose
the standard OpenEnv endpoints and adds several competition‑style routes.

### Core endpoints

These are created automatically by `create_fastapi_app`:

- `GET /health`
  - Liveness probe; returns a simple JSON payload when the server is healthy.

- `POST /reset`
  - Body: empty JSON `{}`.
  - Effect: calls `ICUEnvironment.reset()` and returns the initial `ICUObservation`.

- `POST /step`
  - Body: an `ICUAction` JSON document, e.g.:

    ```json
    {
      "action_type": "ASSIGN_BED",
      "patient_id": "PT-01",
      "bed_id": "S1"
    }
    ```

    or

    ```json
    {
      "action_type": "STEP_DOWN",
      "patient_id": "PT-04"
    }
    ```

  - Effect: advances the environment one step and returns the new observation.

- `GET /state`
  - Returns a JSON view of `env.state` (unassigned patients, active patients, wards).

- `/docs`
  - Swagger / OpenAPI UI built by FastAPI.

### Additional endpoints in app2

- `GET /`
  - Summary of the v2 environment and a list of important endpoints (`/health`, `/docs`, `/tasks`, `/grader`, `/baseline`, `/web`).

- `GET /web`
  - Lightweight HTML UI for manual testing:
    - Buttons for `/health`, `/reset`, `/state`, `/tasks`, `/grader`, `/baseline`.
    - A textarea to send arbitrary `ICUAction` JSON to `/step`.

- `GET /tasks`
  - Returns a small JSON description of the available task and the action schema
    (how to format `ASSIGN_BED` and `STEP_DOWN` requests).

- `POST /grader?steps=N`
  - Runs a **random baseline** policy for up to `N` steps:
    - Randomly assigns waiting patients to empty beds.
    - Otherwise randomly steps down stable ICU patients.
  - Response JSON includes:
    - `score`, `total_reward`, `steps`, `fatal_errors`, `feedback`.

- `GET /baseline`
  - Runs a more **heuristic baseline** policy that:
    - Tries to assign patients only to clinically feasible beds.
    - Falls back to safe step‑down of long‑stay stable patients.
  - Response JSON includes the same metrics plus a `baseline_agent` label.

The `main()` function in app2 starts the server with Uvicorn on port `8001`:

- `uvicorn.run("server.app2:app", host="0.0.0.0", port=8001, workers=1)`.

---

## Grader v2 (Python, no HTTP)

The file [test_env/grader_v2.py](test_env/grader_v2.py) runs the environment
directly in‑process, without going through HTTP.

- It imports `ICUEnvironment` from `test_env.server.env2` and the v2 models.
- Defines a `Policy` protocol with `act(env, obs) -> Optional[ICUAction]`.
- Provides two baseline policies:
  - `RandomAssignPolicy`: random feasible assignments, then random safe step‑downs.
  - `HeuristicSafePolicy`: constraint‑aware assignment and safe step‑down.
- The `run_episode` helper:
  - Calls `env.reset()`, loops over `policy.act(env, obs)` and `env.step(action)`.
  - Accumulates `total_reward` and stops on `done`, max steps, or when the policy returns `None`.
  - Returns `(score, total_reward, steps, fatal_errors)` from observation metadata.
- The `evaluate` function:
  - Runs multiple episodes and prints averages for score, reward, episode length, and fatal errors.

Running the script directly executes both policies and prints a small report.

---

## Setup Instructions

You need Python 3.11+ and a virtual environment.

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

pip install -r requirements.txt
```

This installs FastAPI, Uvicorn, Pydantic, OpenEnv and related dependencies
used by the environment and server.

---

## Running the v2 FastAPI Server (env2 + app2)

1. **Activate the virtualenv and install deps** (see above).

2. **Start the server** from the `test_env` folder:

   ```bash
   cd test_env
   python -m server.app2
   ```

   This will start Uvicorn on `http://localhost:8001` using `server/app2.py`.

3. **Explore the API**:
   - Open the docs: `http://localhost:8001/docs`
   - Use the web UI: `http://localhost:8001/web`

4. **Example HTTP calls** (from another terminal):

   ```bash
   # Reset the environment
   curl -X POST http://localhost:8001/reset

   # Assign patient PT-01 to ventilator bed S1
   curl -X POST http://localhost:8001/step \
              -H "Content-Type: application/json" \
              -d "{\"action_type\": \"ASSIGN_BED\", \"patient_id\": \"PT-01\", \"bed_id\": \"S1\"}"

   # Step-down stable patient PT-04
   curl -X POST http://localhost:8001/step \
              -H "Content-Type: application/json" \
              -d "{\"action_type\": \"STEP_DOWN\", \"patient_id\": \"PT-04\"}"

   # Inspect simulator state
   curl http://localhost:8001/state

   # Run the random grader for 50 steps
   curl -X POST "http://localhost:8001/grader?steps=50"

   # Run the heuristic baseline
   curl http://localhost:8001/baseline
   ```

---

## Running grader_v2 (direct Python evaluation)

You can also evaluate the environment without HTTP using
[test_env/grader_v2.py](test_env/grader_v2.py).

1. **Activate the virtualenv and install deps** (same as above).

2. **Run the grader from the repo root**:

   ```bash
   python test_env/grader_v2.py
   ```

   or, using module syntax:

   ```bash
   python -m test_env.grader_v2
   ```

3. The script will:
   - Run `RandomAssignPolicy` for several episodes.
   - Run `HeuristicSafePolicy` for several episodes.
   - Print average score, reward, episode length, and fatal error count.

This is a convenient way to iterate on the environment logic locally
before wiring it into larger RL agents or hosting it via the HTTP API.
