from typing import List, Dict
from openenv.core.env_server.interfaces import Environment

# Import the schemas we defined in the other file
from models2 import (
    PatientState, BedState, WardState, 
    AssignBedAction, StepDownAction, ICUAction, ICUObservation
)

class ICUEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.max_steps = 20
        self.reset()

    # ---------------- RESET ----------------

    def reset(self):
        self.time = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.fatal_errors = 0

        self.active_patients: Dict[str, PatientState] = {}

        self.unassigned_patients: List[PatientState] = [
            PatientState(patient_id="PT-01", age=60, gcs_score=8, needs_ventilator=True,
                         is_infectious=False, has_severe_head_injury=False,
                         has_paralysis=False, days_in_icu=0),
            PatientState(patient_id="PT-02", age=30, gcs_score=12, needs_ventilator=False,
                         is_infectious=True, has_severe_head_injury=False,
                         has_paralysis=False, days_in_icu=0),
            PatientState(patient_id="PT-03", age=45, gcs_score=10, needs_ventilator=False,
                         is_infectious=False, has_severe_head_injury=False,
                         has_paralysis=True, days_in_icu=0),
        ]

        # Dummy patient to test Step-Down functionality
        dummy_stable = PatientState(patient_id="PT-04", age=50, gcs_score=15, needs_ventilator=False,
                                    is_infectious=False, has_severe_head_injury=False,
                                    has_paralysis=False, days_in_icu=5)
        self.active_patients["PT-04"] = dummy_stable

        self.wards = [
            WardState(ward_name="Standard ICU", beds=[
                BedState(bed_id="S1", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=False),
                BedState(bed_id="S2", has_ventilator=False, is_negative_pressure=True, has_specialized_mattress=False),
                BedState(bed_id="S3", has_ventilator=False, is_negative_pressure=False, has_specialized_mattress=True),
                BedState(bed_id="S4", has_ventilator=False, is_negative_pressure=False, has_specialized_mattress=False, current_occupant_id="PT-04")
            ])
        ]

        return self._get_obs(0.0)

    # ---------------- STEP ----------------

    def step(self, action: ICUAction):
        if self.done:
            return self._get_obs(0.0)

        self.time += 1
        
        # Dense Reward: Dynamic Queue Tax
        reward = -0.02 * len(self.unassigned_patients)

        # Deterministic Deterioration
        if self.time % 3 == 0:
            for p in self.unassigned_patients:
                if p.gcs_score > 3:
                    p.gcs_score -= 1
        
        # Patient Recovery Time
        for p in self.active_patients.values():
            p.days_in_icu += 1

        try:
            if action.action_type == "ASSIGN_BED":
                reward += self._assign(action)
            elif action.action_type == "STEP_DOWN":
                reward += self._step_down(action)

        except Exception as e:
            # Fatal Clinical Error Penalty
            reward -= 0.5
            self.fatal_errors += 1
            feedback = f"ACTION FAILED: {str(e)}"
            self.done = True 
        else:
            feedback = f"Action successful: {action.action_type} for {action.patient_id}."

        self.cumulative_reward += reward

        # Win condition
        if not self.unassigned_patients and self.fatal_errors == 0:
            reward += 1.0
            self.done = True
            feedback = "SUCCESS: Shift completed safely."

        if self.time >= self.max_steps:
            self.done = True
            feedback = "FAILED: Shift ended before all patients were triaged."

        return self._get_obs(reward, feedback)

    # ---------------- ACTIONS & CLINICAL GRADERS ----------------

    def _assign(self, action: AssignBedAction):
        patient = next((p for p in self.unassigned_patients if p.patient_id == action.patient_id), None)
        if not patient:
            raise ValueError(f"Patient {action.patient_id} not found in waiting queue.")

        bed = next((b for w in self.wards for b in w.beds if b.bed_id == action.bed_id), None)
        if not bed:
            raise ValueError(f"Bed {action.bed_id} does not exist.")
        if bed.current_occupant_id is not None:
            raise ValueError(f"Bed {action.bed_id} is already occupied.")

        # Constraints Grader
        if patient.needs_ventilator and not bed.has_ventilator:
            raise ValueError("Clinical Error: Patient requires a ventilator. Target bed lacks one.")
        if patient.is_infectious and not bed.is_negative_pressure:
            raise ValueError("Clinical Error: Patient is infectious. Target bed lacks negative pressure isolation.")
        if patient.has_paralysis and not bed.has_specialized_mattress:
            raise ValueError("Clinical Error: Patient has paralysis. Target bed lacks a specialty mattress.")

        # Execute Assignment
        bed.current_occupant_id = action.patient_id
        self.unassigned_patients.remove(patient)
        self.active_patients[patient.patient_id] = patient

        # Dense Reward: Urgency Bonus
        base_reward = 0.2
        urgency_bonus = (15 - patient.gcs_score) * 0.02
        vent_bonus = 0.1 if patient.needs_ventilator else 0.0
        
        return base_reward + urgency_bonus + vent_bonus

    def _step_down(self, action: StepDownAction):
        patient = self.active_patients.get(action.patient_id)
        if not patient:
            raise ValueError(f"Patient {action.patient_id} is not currently in a bed.")

        # Constraints Grader
        if patient.gcs_score < 14:
            raise ValueError("Clinical Error: Patient consciousness too low for general ward step-down.")
        if patient.needs_ventilator:
            raise ValueError("Clinical Error: Cannot step down a ventilator-dependent patient.")

        # Execute Step Down
        bed = next((b for w in self.wards for b in w.beds if b.current_occupant_id == action.patient_id), None)
        if bed:
            bed.current_occupant_id = None
        
        del self.active_patients[action.patient_id]

        # Dense Reward: Efficiency Bonus
        base_reward = 0.2
        days_past_recovery = max(0, patient.days_in_icu - 3)
        efficiency_bonus = days_past_recovery * 0.05
        
        return base_reward + efficiency_bonus

    # ---------------- OBSERVATION & UTILS ----------------

    def _summary(self):
        beds = []
        for w in self.wards:
            for b in w.beds:
                if b.current_occupant_id is None:
                    beds.append(b.bed_id)
        return f"Available empty bed IDs: {', '.join(beds) if beds else 'None'}"

    def _get_obs(self, reward, feedback="Environment Initialized."):
        score = 0.0 if self.fatal_errors > 0 else max(0.0, min(1.0, self.cumulative_reward))

        return ICUObservation(
            hospital_summary=self._summary(),
            unassigned_patients=self.unassigned_patients,
            feedback=feedback,
            reward=reward,
            done=self.done,
            metadata={
                "score": score,
                "step": self.time,
                "fatal_errors": self.fatal_errors
            }
        )

    @property
    def state(self):
        return {
            "unassigned_patients": self.unassigned_patients,
            "active_patients": self.active_patients,
            "wards": self.wards
        }