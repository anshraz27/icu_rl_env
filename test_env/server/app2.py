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

# Fix import path so "test_env" and siblings are importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models2 import (  # type: ignore
	ICUAction,
	ICUObservation,
	PatientState,
	BedState,
	WardState,
	AssignBedAction,
	StepDownAction,
)
from server.env2 import ICUEnvironment  # type: ignore


# ---------------- AUTO ENV ROUTES ----------------
# This creates /reset, /step, /state, /health, /docs

app = create_fastapi_app(ICUEnvironment, ICUAction, ICUObservation)


# ---------------- ROOT ----------------

@app.get("/")
def root():
	return JSONResponse(
		content={
			"name": "ICU Bed Allocation Environment v2",
			"version": "1.0.0",
			"status": "running",
			"description": "Structured ICU triage environment with bed assignment and step-down logic.",
			"endpoints": {
				"health": "/health",
				"docs": "/docs",
				"tasks": "/tasks",
				"grader": "/grader",
				"baseline": "/baseline",
				"web": "/web",
			},
		}
	)


# ---------------- WEB UI ----------------

@app.get("/web", tags=["UI"], response_class=HTMLResponse)
def web_ui():
	# Lightweight UI focused on showing how to call /reset, /step, /state
	html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
	<meta charset=\"UTF-8\" />
	<title>ICU Env v2 Web UI</title>
	<style>
		body { font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 960px; }
		h1 { margin-bottom: 0.25rem; }
		h2 { margin-top: 1.5rem; }
		section { border: 1px solid #e5e7eb; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-top: 0.75rem; }
		label { display: inline-block; min-width: 120px; }
		input, select, textarea { padding: 0.2rem 0.4rem; }
		button { margin-top: 0.5rem; padding: 0.25rem 0.75rem; cursor: pointer; }
		pre { background: #111827; color: #e5e7eb; padding: 0.75rem; border-radius: 0.5rem; max-height: 320px; overflow: auto; }
		code { background: #f3f4f6; padding: 0 0.2rem; border-radius: 0.25rem; }
		textarea { width: 100%; box-sizing: border-box; min-height: 80px; }
	</style>
</head>
<body>
	<h1>ICU Bed Allocation Environment v2</h1>
	<p>Interactive web UI for the v2 environment using structured ICUAction payloads.</p>

	<section>
		<h2>Core environment endpoints</h2>
		<div>
			<button onclick=\"callHealth()\">GET /health</button>
			<button onclick=\"callReset()\">POST /reset</button>
			<br />
			<h3>Step with ICUAction</h3>
			<p>Provide a JSON ICUAction, e.g. <code>{&quot;action_type&quot;: &quot;ASSIGN_BED&quot;, &quot;patient_id&quot;: &quot;PT-01&quot;, &quot;bed_id&quot;: &quot;S1&quot;}</code></p>
			<textarea id=\"step-body\">{\n  \"action_type\": \"ASSIGN_BED\",\n  \"patient_id\": \"PT-01\",\n  \"bed_id\": \"S1\"\n}</textarea>
			<button onclick=\"callStep()\">POST /step</button>
			<button onclick=\"callState()\">GET /state</button>
		</div>
	</section>

	<section>
		<h2>Competition endpoints</h2>
		<div>
			<button onclick=\"callTasks()\">GET /tasks</button>
			<br />
			<label>Grader steps:</label>
			<input id=\"grader-steps\" type=\"number\" min=\"1\" value=\"100\" />
			<button onclick=\"callGrader()\">POST /grader?steps=...</button>
			<br />
			<button onclick=\"callBaseline()\">GET /baseline</button>
		</div>
	</section>

	<section>
		<h2>Docs</h2>
		<p>
			OpenAPI docs are available at <code>/docs</code>.
			<button onclick=\"openDocs()\">Open /docs in new tab</button>
		</p>
	</section>

	<h2>Last response</h2>
	<pre id=\"output\">(no requests yet)</pre>

	<script>
		function showResponse(label, res, bodyText) {
			document.getElementById('output').textContent =
				label + '\\nStatus: ' + res.status + '\\n\\n' + bodyText;
		}

		async function callHealth() {
			const res = await fetch('/health');
			const txt = await res.text();
			showResponse('GET /health', res, txt);
		}

		async function callReset() {
			const res = await fetch('/reset', { method: 'POST' });
			const txt = await res.text();
			showResponse('POST /reset', res, txt);
		}

		async function callStep() {
			const body = document.getElementById('step-body').value;
			const res = await fetch('/step', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body
			});
			const txt = await res.text();
			showResponse('POST /step (ICUAction JSON)', res, txt);
		}

		async function callState() {
			const res = await fetch('/state');
			const txt = await res.text();
			showResponse('GET /state', res, txt);
		}

		async function callTasks() {
			const res = await fetch('/tasks');
			const txt = await res.text();
			showResponse('GET /tasks', res, txt);
		}

		async function callGrader() {
			const stepsInput = document.getElementById('grader-steps');
			const steps = parseInt(stepsInput.value || '100', 10);
			const res = await fetch('/grader?steps=' + encodeURIComponent(steps), {
				method: 'POST'
			});
			const txt = await res.text();
			showResponse('POST /grader?steps=' + steps, res, txt);
		}

		async function callBaseline() {
			const res = await fetch('/baseline');
			const txt = await res.text();
			showResponse('GET /baseline', res, txt);
		}

		function openDocs() {
			window.open('/docs', '_blank');
		}
	</script>
</body>
</html>"""
	return HTMLResponse(content=html)


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

		action: ICUAction

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

	score = float(obs.metadata.get("score", 0.0) or 0.0)

	return JSONResponse(
		content={
			"score": score,
			"total_reward": total_reward,
			"steps": int(obs.metadata.get("step", 0)),
			"fatal_errors": int(obs.metadata.get("fatal_errors", 0)),
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

		action: ICUAction | None = None

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

	score = float(obs.metadata.get("score", 0.0) or 0.0)

	return JSONResponse(
		content={
			"baseline_agent": "v2_heuristic (feasible bed first, then safe step-down)",
			"score": score,
			"total_reward": total_reward,
			"steps": int(obs.metadata.get("step", 0)),
			"fatal_errors": int(obs.metadata.get("fatal_errors", 0)),
			"feedback": obs.feedback,
		}
	)


# ---------------- RUN SERVER ----------------

def main():
	import uvicorn

	port = int(os.environ.get("PORT", 8001))
	host = os.environ.get("HOST", "0.0.0.0")

	# Note: using app2:app so it doesn't clash with server.app
	uvicorn.run("server.app2:app", host=host, port=port, workers=1)


if __name__ == "__main__":
	main()

