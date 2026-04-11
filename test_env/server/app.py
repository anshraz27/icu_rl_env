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

			},
		}
	)




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

