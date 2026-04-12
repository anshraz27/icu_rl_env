"""
server/app2.py - FastAPI server for ICU Environment v2 (bed / step-down logic).

This app mirrors the endpoints from server/app.py but is wired to the
new environment logic defined in env2.py and schemas in models2.py.
"""

import os
import sys
from typing import List
from fastapi.responses import JSONResponse, HTMLResponse
from openenv.core.env_server import create_fastapi_app
from dotenv import load_dotenv
load_dotenv(override=True)
# Fix import path so "test_env" and siblings are importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (  # type: ignore
	ICUActionRouter,
	ICUObservation,
	PatientState,
	BedState,
	WardState,
	AssignBedAction,
	StepDownAction,
)
from server.env2 import ICUEnvironment  


# ---------------- AUTO ENV ROUTES ----------------
# This creates /reset, /step, /state, /health, /docs

app = create_fastapi_app(ICUEnvironment, ICUActionRouter, ICUObservation)

# ---------------- ROOT ----------------
import textwrap
from fastapi.responses import HTMLResponse

@app.get("/web")
def web():
    html_content = textwrap.dedent("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ICU Dashboard</title>
        <style>
            body {
                font-family: Arial;
                background: #f5f7fa;
                margin: 20px;
            }

            h1 {
                color: #2c3e50;
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }

            .card {
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }

            .patient {
                border-left: 5px solid #3498db;
                padding: 10px;
                margin: 10px 0;
                background: #ecf5ff;
            }

            .critical {
                border-left-color: red;
                background: #ffecec;
            }

            .medium {
                border-left-color: orange;
            }

            button {
                padding: 8px 12px;
                margin-top: 5px;
                border: none;
                background: #3498db;
                color: white;
                border-radius: 5px;
                cursor: pointer;
            }

            button:hover {
                background: #2980b9;
            }

            input {
                padding: 5px;
                margin: 5px 0;
            }

            pre {
                background: #1e1e1e;
                color: #00ffcc;
                padding: 10px;
                border-radius: 5px;
                max-height: 200px;
                overflow: auto;
            }
        </style>
    </head>

    <body>

        <h1>🏥 ICU Bed Allocation Dashboard</h1>

        <button onclick="resetEnv()">🔄 Reset</button>

        <div class="grid">
            <div class="card">
                <h2>🧍 Patients</h2>
                <div id="patients"></div>
            </div>

            <div class="card">
                <h2>🛏️ Available Beds</h2>
                <div id="beds"></div>
            </div>
        </div>

		<div class="card">
			<h2>📊 Logs</h2>
			<pre id="logs">Waiting...</pre>
		</div>

		<div class="card">
			<h2>🔌 API Controls</h2>

			<button onclick="callResetApi()">Call /reset</button>
			<button onclick="callStepApi()">Call /step</button>
			<button onclick="callStateApi()">Call /state</button>

			<h3>/reset response</h3>
			<pre id="reset-response">Not called yet</pre>

			<h3>/step response</h3>
			<pre id="step-response">Not called yet</pre>

			<h3>/state response</h3>
			<pre id="state-response">Not called yet</pre>
		</div>

		<div class="card">
			<h2>🏁 Run Grader</h2>

			<label>Steps:</label>
			<input type="number" id="grader-steps" value="50" />

			<br>
			<button onclick="runGrader()">Run Evaluation</button>

			<h3>Result:</h3>
			<pre id="grader-output">Not run yet</pre>
		</div>

        <script>
            let currentObs = null;

            async function resetEnv() {
                const res = await fetch('/reset', { method: 'POST' });
                const data = await res.json();
                updateUI(data.observation);
            }

            async function step(action) {
                const res = await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ action })   // FIXED
                });
                const data = await res.json();
                updateUI(data.observation);
            }

            function updateUI(obs) {
                currentObs = obs;

                // Patients
                const patientsDiv = document.getElementById('patients');
                patientsDiv.innerHTML = "";

                obs.unassigned_patients.forEach((p) => {
                    let cls = "patient";
                    if (p.gcs_score < 8) cls += " critical";
                    else if (p.gcs_score < 12) cls += " medium";

                    patientsDiv.innerHTML += `
                        <div class="${cls}">
                            <b>ID:</b> ${p.patient_id}<br>
                            GCS: ${p.gcs_score} | Vent: ${p.needs_ventilator}<br>
                            <button onclick="assign('${p.patient_id}')">Assign</button>
                        </div>
                    `;
                });

                // Beds (show full summary)
                document.getElementById('beds').innerText = obs.hospital_summary;

                // Logs
                document.getElementById('logs').innerText =
                    JSON.stringify(obs, null, 2);
            }

            function assign(patient_id) {
                step({
                    action_type: "ASSIGN_BED",
                    patient_id: patient_id,
                    bed_id: "S1"   // TODO: make dynamic later
                });
            }

            function stepDown(patient_id) {
                step({
                    action_type: "STEP_DOWN",
                    patient_id: patient_id
                });
            }

            async function runGrader() {
                const steps = document.getElementById("grader-steps").value;

                const res = await fetch(`/grader?steps=${steps}`, {
                    method: "POST"
                });

                const data = await res.json();

                document.getElementById("grader-output").innerText =
                    JSON.stringify(data, null, 2);
            }

			// ---- Explicit API call helpers ----

			async function callResetApi() {
				try {
					const res = await fetch('/reset', { method: 'POST' });
					const data = await res.json();
					document.getElementById('reset-response').innerText =
						JSON.stringify(data, null, 2);
					if (data.observation) {
						updateUI(data.observation);
					}
				} catch (err) {
					document.getElementById('reset-response').innerText = String(err);
				}
			}

			async function callStepApi() {
				try {
					if (!currentObs || !currentObs.unassigned_patients || currentObs.unassigned_patients.length === 0) {
						document.getElementById('step-response').innerText =
							'No unassigned patients available to create a sample /step action.';
						return;
					}

					const patient = currentObs.unassigned_patients[0];
					const action = {
						action_type: "ASSIGN_BED",
						patient_id: patient.patient_id,
						bed_id: "S1"
					};

					const res = await fetch('/step', {
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ action })
					});
					const data = await res.json();
					document.getElementById('step-response').innerText =
						JSON.stringify(data, null, 2);
					if (data.observation) {
						updateUI(data.observation);
					}
				} catch (err) {
					document.getElementById('step-response').innerText = String(err);
				}
			}

			async function callStateApi() {
				try {
					const res = await fetch('/state');
					const data = await res.json();
					document.getElementById('state-response').innerText =
						JSON.stringify(data, null, 2);
				} catch (err) {
					document.getElementById('state-response').innerText = String(err);
				}
			}
        </script>

    </body>
    </html>
    """)
    return HTMLResponse(content=html_content)




# ---------------- TASKS ----------------

@app.get("/tasks", tags=["Competition"])
def get_tasks():
	return JSONResponse(
		content={
			"tasks": [
				{
					"task_id": "icu_v2_simple",
					"description": "Simple mixed case load with bed assignment and step-down.",
					"difficulty": "medium",
				},
			],
			"total": 1,
			"action_schema": {
				"ASSIGN_BED": {
					"action_type": "ASSIGN_BED",
					"patient_id": "str (e.g. PT-01)",
					"bed_id": "str (e.g. S1)",
				},
				"STEP_DOWN": {
					"action_type": "STEP_DOWN",
					"patient_id": "str (e.g. PT-04)",
				},
			},
		}
	)


# ---------------- HELPERS FOR GRADER / BASELINE ----------------

def _list_empty_beds(wards: List[WardState]) -> List[BedState]:
	return [
		bed
		for ward in wards
		for bed in ward.beds
		if bed.current_occupant_id is None
	]


# ---------------- GRADER ----------------

@app.post("/grader", tags=["Competition"])
def run_grader(steps: int = 100):
	"""Runs a simple random baseline on the v2 environment.

	Randomly assigns waiting patients to empty beds when possible,
	otherwise attempts random safe step-downs. Returns score, reward,
	and basic metadata derived from ICUObservation.metadata.
	"""

	import random

	env = ICUEnvironment()
	obs: ICUObservation = env.reset()

	total_reward = 0.0

	for _ in range(steps):
		if obs.done:
			break

		state = env.state
		unassigned: List[PatientState] = state["unassigned_patients"]
		active = state["active_patients"]
		wards: List[WardState] = state["wards"]

		empty_beds = _list_empty_beds(wards)

		action: ICUActionRouter

		# Randomly assign if there is someone waiting and free beds
		if unassigned and empty_beds:
			patient = random.choice(unassigned)
			bed = random.choice(empty_beds)
			action = AssignBedAction(
				action_type="ASSIGN_BED",
				patient_id=patient.patient_id,
				bed_id=bed.bed_id,
			)
		else:
			# Otherwise, try a random safe-ish step-down
			candidates = [
				p for p in active.values() if (p.gcs_score >= 14 and not p.needs_ventilator)
			]
			if candidates:
				patient = random.choice(candidates)
				action = StepDownAction(
					action_type="STEP_DOWN",
					patient_id=patient.patient_id,
				)
			else:
				break

		obs = env.step(action)
		total_reward += obs.reward

	score = float(obs.score)

	return JSONResponse(
		content={
			"score": score,
			"total_reward": total_reward,
			"steps": obs.step,
			"fatal_errors": obs.fatal_errors,
			"feedback": obs.feedback,
		}
	)


# ---------------- BASELINE ----------------

@app.get("/baseline", tags=["Competition"])
def run_baseline():
	"""Heuristic baseline using the v2 clinical rules.

	- Tries to assign each waiting patient to the first feasible empty bed
	  that satisfies all clinical constraints.
	- If no assignments are possible but a stable ICU patient can be
	  stepped down, it steps down the longest-stay stable patient.
	"""

	env = ICUEnvironment()
	obs: ICUObservation = env.reset()

	total_reward = 0.0

	while not obs.done:
		state = env.state
		unassigned: List[PatientState] = state["unassigned_patients"]
		active = state["active_patients"]
		wards: List[WardState] = state["wards"]

		empty_beds = _list_empty_beds(wards)

		action: ICUActionRouter 

		# 1) Try to find a feasible assignment first
		if unassigned and empty_beds:
			for patient in unassigned:
				for bed in empty_beds:
					if patient.needs_ventilator and not bed.has_ventilator:
						continue
					if patient.is_infectious and not bed.is_negative_pressure:
						continue
					if patient.has_paralysis and not bed.has_specialized_mattress:
						continue

					action = AssignBedAction(
						action_type="ASSIGN_BED",
						patient_id=patient.patient_id,
						bed_id=bed.bed_id,
					)
					break
				if action is not None:
					break

		# 2) If no assignment, attempt a safe step-down
		if action is None:
			candidates = [
				p
				for p in active.values()
				if (p.gcs_score >= 14 and not p.needs_ventilator)
			]
			if candidates:
				candidates.sort(key=lambda p: p.days_in_icu, reverse=True)
				patient = candidates[0]
				action = StepDownAction(
					action_type="STEP_DOWN",
					patient_id=patient.patient_id,
				)

		# If nothing to do, break the loop
		if action is None:
			break

		obs = env.step(action)
		total_reward += obs.reward

	score = float(obs.score)

	return JSONResponse(
		content={
			"baseline_agent": "v2_heuristic (feasible bed first, then safe step-down)",
			"score": score,
			"total_reward": total_reward,
			"steps": obs.step,
			"fatal_errors": obs.fatal_errors,
			"feedback": obs.feedback,
		}
	)


# ---------------- RUN SERVER ----------------

def main():
	import uvicorn

	port = int(os.environ.get("PORT", 7860))
	host = os.environ.get("HOST", "0.0.0.0")

	# Note: using app2:app so it doesn't clash with server.app
	uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
	main()

