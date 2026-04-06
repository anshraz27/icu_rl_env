import random
from uuid import uuid4
from typing import List

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ICUAction, ICUObservation, ICUState, Patient
except ImportError:
    from models import ICUAction, ICUObservation, ICUState, Patient


class ICUEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.max_time = 50
        self.max_patients = 20
        self.task_counter = 0
        self.reset()

    # ---------------- RESET ----------------

    def reset(self) -> ICUObservation:

        self.task_counter += 1
        self.task_level = ((self.task_counter - 1) % 3) + 1

        if self.task_level == 1:
            self.bed_capacity = 3
            self.arrival_prob = 0.3
            self.severity_min, self.severity_max = 1, 3
        elif self.task_level == 2:
            self.bed_capacity = 2
            self.arrival_prob = 0.5
            self.severity_min, self.severity_max = 2, 4
        else:
            self.bed_capacity = 1
            self.arrival_prob = 0.7
            self.severity_min, self.severity_max = 3, 5

        self.time = 0
        self.available_beds = self.bed_capacity

        self.patients: List[Patient] = []
        self.in_treatment = []

        self.patient_id_counter = 0
        self.total_patients = 0
        self.total_admitted = 0
        self.deaths = 0

        # tracking for scoring
        self.history = []
        self.total_queue_time = 0
        self.bad_decisions = 0

        self._state = ICUState(
            episode_id=str(uuid4()),
            step_count=0,
            total_patients=0,
            total_admitted=0,
            deaths=0,
        )

        return self._get_obs(0.0, False)

    # ---------------- GRADER (FINAL SCORE) ----------------

    def grade_state(self):
        if self.total_patients == 0:
            return 0.0, "No patients yet"

        survival_rate = (self.total_patients - self.deaths) / self.total_patients

        critical_total = sum(1 for h in self.history if h["severity"] >= 4)
        critical_saved = sum(
            1 for h in self.history if h["severity"] >= 4 and h["saved"]
        )

        critical_rate = critical_saved / max(1, critical_total)

        avg_queue = self.total_queue_time / max(1, self.time)
        efficiency = max(0.0, 1.0 - (avg_queue / 10))

        mistake_penalty = self.bad_decisions / max(1, self.total_admitted)

        score = (
            0.4 * survival_rate +
            0.3 * critical_rate +
            0.2 * efficiency -
            0.1 * mistake_penalty
        )

        score = max(0.0, min(1.0, score))

        feedback = (
            f"Survival: {survival_rate:.2f} | "
            f"Critical: {critical_rate:.2f} | "
            f"Efficiency: {efficiency:.2f} | "
            f"Mistakes: {mistake_penalty:.2f}"
        )

        return score, feedback

    # ---------------- STEP ----------------

    def step(self, action: ICUAction) -> ICUObservation:

        reward = 0.0

        # 1. arrivals
        if len(self.patients) < self.max_patients and random.random() < self.arrival_prob:
            p = Patient(
                id=self.patient_id_counter,
                severity=random.randint(self.severity_min, self.severity_max),
                wait_time=0,
                survival_prob=1.0,
            )
            self.patient_id_counter += 1
            self.patients.append(p)
            self.total_patients += 1

        # 2. update patients
        for p in self.patients:
            p.wait_time += 1
            p.survival_prob = max(0.0, p.survival_prob - 0.05)

        # 3. queue penalty
        reward -= 0.02 * len(self.patients)
        self.total_queue_time += len(self.patients)

        # 4. action handling
        if action.value > 0:
            idx = action.value - 1

            if 0 <= idx < len(self.patients) and self.available_beds > 0:
                patient = self.patients.pop(idx)

                # reward for good decision
                if patient.severity >= 4:
                    reward += 1.0
                elif patient.severity >= 2:
                    reward += 0.5
                else:
                    reward -= 0.2

                # penalty for wrong prioritization
                if patient.severity <= 2 and any(p.severity >= 4 for p in self.patients):
                    reward -= 0.5
                    self.bad_decisions += 1

                self.available_beds -= 1
                self.total_admitted += 1

                self.history.append({
                    "severity": patient.severity,
                    "saved": True
                })

                self.in_treatment.append({
                    "remaining": random.randint(2, 5)
                })

            else:
                reward -= 0.3
                self.bad_decisions += 1

        else:
            if self.patients:
                reward -= 0.1

        # 5. treatment progression
        new_treatment = []
        for t in self.in_treatment:
            t["remaining"] -= 1
            if t["remaining"] <= 0:
                self.available_beds += 1
            else:
                new_treatment.append(t)
        self.in_treatment = new_treatment

        # 6. deaths
        survivors = []
        for p in self.patients:
            if p.survival_prob <= 0:
                if p.severity >= 4:
                    reward -= 1.5
                else:
                    reward -= 0.5

                self.history.append({
                    "severity": p.severity,
                    "saved": False
                })
                self.deaths += 1
            else:
                survivors.append(p)

        self.patients = survivors

        # 7. update state
        self.time += 1
        self._state.step_count += 1
        self._state.total_patients = self.total_patients
        self._state.total_admitted = self.total_admitted
        self._state.deaths = self.deaths

        done = self.time >= self.max_time or self.total_patients >= self.max_patients

        # score is separate
        score, feedback = self.grade_state()

        obs = self._get_obs(reward, done)

        if done:
            obs.metadata["score"] = score

        return obs

    # ---------------- OBS ----------------

    def _get_obs(self, reward, done):
        score, feedback = self.grade_state()

        hint = ""
        if self.time > 10 and score < 0.5:
            hint = "Try prioritizing high severity patients"

        return ICUObservation(
            patients=list(self.patients),
            available_beds=self.available_beds,
            time_step=self.time,
            reward=reward,
            done=done,
            metadata={
                "task_level": self.task_level,
                "score": score,
                "feedback": feedback,
                "hint": hint
            },
        )

    @property
    def state(self):
        return self._state