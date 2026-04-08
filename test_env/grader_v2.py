"""Grader / evaluator for the v2 ICU Bed Allocation environment.

This grader exercises the new logic implemented in
- test_env/server/env2.py
- test_env/models2.py

It runs a few baseline policies directly against ICUEnvironment
(without using the HTTP client) and reports average score and reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, Optional, List
import sys
from pathlib import Path
import random

# ---------------- SETUP ----------------

random.seed(42)

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from test_env.server.env2 import ICUEnvironment
from test_env.models import (
    ICUAction,
    ICUObservation,
    PatientState,
    BedState,
    WardState,
    AssignBedAction,
    StepDownAction,
)


# ---------------- POLICY INTERFACE ----------------

class Policy(Protocol):
    def act(self, env: ICUEnvironment, obs: ICUObservation) -> Optional[ICUAction]:
        """Return an ICUAction or None to indicate no safe action.

        Returning None allows the episode loop to terminate early
        when the policy believes no further safe / meaningful
        actions are available.
        """


# ---------------- BASELINE POLICIES ----------------

@dataclass
class RandomAssignPolicy:
    """Very simple random policy.

    - If there are unassigned patients and at least one empty bed,
      randomly assign a waiting patient to a random empty bed.
    - Otherwise, if there is a patient in ICU that looks safe to step down
      (GCS >= 14 and not ventilator-dependent), step them down.
    - If neither is possible, return None to stop the episode.
    """

    def act(self, env: ICUEnvironment, obs: ICUObservation) -> Optional[ICUAction]:
        # Access environment state for beds and active patients
        state = env.state
        unassigned: List[PatientState] = state["unassigned_patients"]
        active = state["active_patients"]  # dict patient_id -> PatientState
        wards: List[WardState] = state["wards"]

        empty_beds: List[BedState] = [
            bed
            for ward in wards
            for bed in ward.beds
            if bed.current_occupant_id is None
        ]

        # Try random assignment first if possible
        if unassigned and empty_beds:
            patient = random.choice(unassigned)
            bed = random.choice(empty_beds)

            return AssignBedAction(
                action_type="ASSIGN_BED",
                patient_id=patient.patient_id,
                bed_id=bed.bed_id,
            )

        # Otherwise attempt a random safe-ish step-down
        stepdown_candidates = [
            p for p in active.values() if (p.gcs_score >= 14 and not p.needs_ventilator)
        ]
        if stepdown_candidates:
            patient = random.choice(stepdown_candidates)
            return StepDownAction(
                action_type="STEP_DOWN",
                patient_id=patient.patient_id,
            )

        # No reasonable action
        return None


@dataclass
class HeuristicSafePolicy:
    """Heuristic policy that tries to respect bed/patient constraints.

    - For each waiting patient, looks for the first empty bed that satisfies
      all hard clinical constraints (ventilator, isolation, mattress).
    - If no such assignment is possible but a stable ICU patient can be
      stepped down safely, performs a step-down.
    - If nothing safe can be done, returns None.
    """

    def act(self, env: ICUEnvironment, obs: ICUObservation) -> Optional[ICUAction]:
        state = env.state
        unassigned: List[PatientState] = state["unassigned_patients"]
        active = state["active_patients"]
        wards: List[WardState] = state["wards"]

        # Helper to iterate empty beds
        def iter_empty_beds() -> List[BedState]:
            return [
                bed
                for ward in wards
                for bed in ward.beds
                if bed.current_occupant_id is None
            ]

        # 1) Try to assign each waiting patient to a feasible bed
        empty_beds = iter_empty_beds()
        if unassigned and empty_beds:
            for patient in unassigned:
                for bed in empty_beds:
                    if patient.needs_ventilator and not bed.has_ventilator:
                        continue
                    if patient.is_infectious and not bed.is_negative_pressure:
                        continue
                    if patient.has_paralysis and not bed.has_specialized_mattress:
                        continue

                    # Found a feasible match
                    return AssignBedAction(
                        action_type="ASSIGN_BED",
                        patient_id=patient.patient_id,
                        bed_id=bed.bed_id,
                    )

        # 2) If no feasible assignment, attempt a safe step-down
        stepdown_candidates = [
            p
            for p in active.values()
            if (p.gcs_score >= 14 and not p.needs_ventilator)
        ]
        if stepdown_candidates:
            # Prefer patients who have stayed longer in ICU
            stepdown_candidates.sort(key=lambda p: p.days_in_icu, reverse=True)
            patient = stepdown_candidates[0]
            return StepDownAction(
                action_type="STEP_DOWN",
                patient_id=patient.patient_id,
            )

        # Nothing safe to do; let the loop terminate
        return None


# ---------------- EPISODE LOOP ----------------

def run_episode(
    env: ICUEnvironment,
    policy: Policy,
    max_steps: int = 100,
) -> Tuple[float, float, int, int]:
    """Run a single episode and return (score, total_reward, steps, fatal_errors)."""

    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while not obs.done and steps < max_steps:
        action = policy.act(env, obs)
        if action is None:
            break

        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps += 1

    # Primary signal: score from metadata (as designed in env2)
    score = float(obs.score)
    fatal_errors = int(obs.fatal_errors)

    return score, total_reward, steps, fatal_errors


# ---------------- EVALUATION ----------------

def evaluate(policy: Policy, episodes: int = 10) -> None:
    scores: List[float] = []
    rewards: List[float] = []
    steps_list: List[int] = []
    error_list: List[int] = []

    env = ICUEnvironment()

    for _ in range(episodes):
        score, reward, steps, fatal_errors = run_episode(env, policy)
        scores.append(score)
        rewards.append(reward)
        steps_list.append(steps)
        error_list.append(fatal_errors)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0
    avg_errors = sum(error_list) / len(error_list) if error_list else 0.0

    print("\nICU Environment v2 Evaluation")
    print("=====================================")
    print(f"Episodes:         {episodes}")
    print(f"Avg Score:        {avg_score:.3f}")
    print(f"Avg Reward:       {avg_reward:.3f}")
    print(f"Avg Episode Steps:{avg_steps:.1f}")
    print(f"Avg Fatal Errors: {avg_errors:.3f}")
    print("=====================================\n")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("Ensure you are using ICUEnvironment from server/env2.py")

    print("\nRunning RandomAssignPolicy...")
    evaluate(RandomAssignPolicy(), episodes=10)

    print("\nRunning HeuristicSafePolicy...")
    evaluate(HeuristicSafePolicy(), episodes=10)
