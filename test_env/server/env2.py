from typing import List, Dict
from openenv.core.env_server.interfaces import Environment
from models import (
    PatientState, BedState, WardState, 
    AssignBedAction, StepDownAction, ICUAction, ICUObservation
)
# ── Score clamping (Borrowed from SQL Env) ──────────────────────────────
def _clamp(score: float, lo: float = 0.00, hi: float = 1.00) -> float:
    """Clamp score strictly to OpenEnv bounds."""
    return max(lo, min(hi, float(score)))

# ── TASK REGISTRY (Borrowed from SQL Env) ───────────────────────────────
# This makes the environment infinitely scalable. Want a new scenario? Just add it here.
TASKS = {
    "easy": {
        "difficulty_level": 1,
        "description": "Direct triage. Beds are plentiful. Assign based on basic constraints.",
        "patients": [
            {"id": "PT-01", "age": 60, "gcs": 8, "vent": True, "inf": False, "head": False, "para": False, "days": 0},
            {"id": "PT-02", "age": 30, "gcs": 12, "vent": False, "inf": True, "head": False, "para": False, "days": 0}
        ],
        "active_occupants": []
    },
    "medium": {
        "difficulty_level": 2,
        "description": "Constraint heavy. Patients have competing specialty needs.",
        "patients": [
            {"id": "PT-03", "age": 45, "gcs": 10, "vent": False, "inf": False, "head": False, "para": True, "days": 0},
            {"id": "PT-04", "age": 22, "gcs": 6, "vent": True, "inf": False, "head": True, "para": False, "days": 0}
        ],
        "active_occupants": []
    },
    "hard": {
        "difficulty_level": 3,
        "description": "Capacity crisis. The agent must step-down a stable patient to free a bed.",
        "patients": [
            {"id": "PT-05", "age": 65, "gcs": 7, "vent": True, "inf": False, "head": False, "para": False, "days": 0}
        ],
        "active_occupants": [
            # This is the "Dummy" patient currently occupying the bed
            {"id": "PT-06", "age": 50, "gcs": 15, "vent": False, "inf": False, "head": False, "para": False, "days": 5, "bed_id": "S1"}
        ]
    }
}

class ICUEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True
    HINT_AFTER_ERRORS = 2 # Progressive Hint Trigger

    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.max_steps = 20
        self.reset()

    # ---------------- RESET (Now completely data-driven) ----------------
    def reset(self):
        # Auto-cycle tasks on every reset call from the client
        if not hasattr(self, 'task_counter'):
            self.task_counter = 0
            
        task_keys = list(TASKS.keys())
        self.task_id = task_keys[self.task_counter % len(task_keys)]
        self.task_counter += 1

        self.time = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.fatal_errors = 0
        self.active_patients: Dict[str, PatientState] = {}
        
        task_data = TASKS[self.task_id]

        # 1. Load Unassigned Patients from Configuration
        self.unassigned_patients = [
            PatientState(patient_id=p["id"], age=p["age"], gcs_score=p["gcs"], needs_ventilator=p["vent"], 
                         is_infectious=p["inf"], has_severe_head_injury=p["head"], has_paralysis=p["para"], days_in_icu=p["days"])
            for p in task_data["patients"]
        ]

        # 2. Build FULL Hospital Layout
        self.wards = [
            WardState(ward_name="PICU", total_beds=2, beds=[
                BedState(bed_id="P1", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=False),
                BedState(bed_id="P2", has_ventilator=False, is_negative_pressure=True, has_specialized_mattress=False)
            ]),
            WardState(ward_name="Isolation ICU", total_beds=2, beds=[
                BedState(bed_id="I1", has_ventilator=True, is_negative_pressure=True, has_specialized_mattress=False),
                BedState(bed_id="I2", has_ventilator=False, is_negative_pressure=True, has_specialized_mattress=False)
            ]),
            WardState(ward_name="Neuro ICU", total_beds=2, beds=[
                BedState(bed_id="N1", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=True),
                BedState(bed_id="N2", has_ventilator=False, is_negative_pressure=False, has_specialized_mattress=False)
            ]),
            WardState(ward_name="Trauma ICU", total_beds=2, beds=[
                BedState(bed_id="T1", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=True),
                BedState(bed_id="T2", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=False)
            ]),
            WardState(ward_name="Standard ICU", total_beds=3, beds=[
                BedState(bed_id="S1", has_ventilator=True, is_negative_pressure=False, has_specialized_mattress=False),
                BedState(bed_id="S2", has_ventilator=False, is_negative_pressure=False, has_specialized_mattress=False),
                BedState(bed_id="S3", has_ventilator=False, is_negative_pressure=False, has_specialized_mattress=True),
            ])
        ]

        # 3. Inject Active Occupants (For Hard Tasks)
        for occ in task_data.get("active_occupants", []):
            pt = PatientState(patient_id=occ["id"], age=occ["age"], gcs_score=occ["gcs"], needs_ventilator=occ["vent"], 
                              is_infectious=occ["inf"], has_severe_head_injury=occ["head"], has_paralysis=occ["para"], days_in_icu=occ["days"])
            self.active_patients[pt.patient_id] = pt
            
            # Assign them to their bed
            for w in self.wards:
                for b in w.beds:
                    if b.bed_id == occ["bed_id"]:
                        b.current_occupant_id = pt.patient_id

        return self._get_obs(0.0)

    # ---------------- STEP LOGIC (Remains the same, just utilizing new _clamp) ----------------
    
    def step(self, action: ICUAction):
        if self.done:
            return self._get_obs(0.0)

        self.time += 1
        reward = -0.02 * len(self.unassigned_patients)

        if self.time % 3 == 0:
            for p in self.unassigned_patients:
                if p.gcs_score > 3: p.gcs_score -= 1
        
        for p in self.active_patients.values():
            p.days_in_icu += 1

        try:
            if action.action_type == "ASSIGN_BED":
                reward += self._assign(action)
            elif action.action_type == "STEP_DOWN":
                reward += self._step_down(action)
        except Exception as e:
            reward -= 0.5
            self.fatal_errors += 1
            feedback = f"ACTION FAILED: {str(e)}"
            # Removed instant-death here to allow the Hint system to work!
        else:
            feedback = f"Action successful: {action.action_type} for {action.patient_id}."

        self.cumulative_reward += reward

        if not self.unassigned_patients and self.fatal_errors == 0:
            reward += 1.0
            self.done = True
            feedback = "SUCCESS: Shift completed safely."

        # Fail state check
        if self.time >= self.max_steps or self.fatal_errors >= 3:
            self.done = True
            feedback = "FAILED: Shift ended or too many clinical errors."

        return self._get_obs(reward, feedback)

    # ---------------- ACTIONS & CLINICAL GRADERS ----------------

    def _assign(self, action: AssignBedAction) -> float:
        patient = next(
            (p for p in self.unassigned_patients if p.patient_id == action.patient_id),
            None,
        )
        if not patient:
            raise ValueError(f"Patient {action.patient_id} not found in waiting queue.")

        bed = next(
            (b for w in self.wards for b in w.beds if b.bed_id == action.bed_id),
            None,
        )
        if not bed:
            raise ValueError(f"Bed {action.bed_id} does not exist.")
        if bed.current_occupant_id is not None:
            raise ValueError(f"Bed {action.bed_id} is already occupied.")

        # Constraints Grader
        if patient.needs_ventilator and not bed.has_ventilator:
            raise ValueError(
                "Clinical Error: Patient requires a ventilator. Target bed lacks one."
            )
        if patient.is_infectious and not bed.is_negative_pressure:
            raise ValueError(
                "Clinical Error: Patient is infectious. Target bed lacks negative pressure isolation."
            )
        if patient.has_paralysis and not bed.has_specialized_mattress:
            raise ValueError(
                "Clinical Error: Patient has paralysis. Target bed lacks a specialty mattress."
            )

        # Execute Assignment
        bed.current_occupant_id = action.patient_id
        self.unassigned_patients.remove(patient)
        self.active_patients[patient.patient_id] = patient

        # Dense Reward: Urgency Bonus
        base_reward = 0.2
        urgency_bonus = (15 - patient.gcs_score) * 0.02
        vent_bonus = 0.1 if patient.needs_ventilator else 0.0

        return base_reward + urgency_bonus + vent_bonus


    def _step_down(self, action: StepDownAction) -> float:
        patient = self.active_patients.get(action.patient_id)
        if not patient:
            raise ValueError(
                f"Patient {action.patient_id} is not currently in a bed."
            )

        # Constraints Grader
        if patient.gcs_score < 14:
            raise ValueError(
                "Clinical Error: Patient consciousness too low for general ward step-down."
            )
        if patient.needs_ventilator:
            raise ValueError(
                "Clinical Error: Cannot step down a ventilator-dependent patient."
            )

        # Execute Step Down
        bed = next(
            (
                b
                for w in self.wards
                for b in w.beds
                if b.current_occupant_id == action.patient_id
            ),
            None,
        )
        if bed:
            bed.current_occupant_id = None

        del self.active_patients[action.patient_id]

        # Dense Reward: Efficiency Bonus
        base_reward = 0.2
        days_past_recovery = max(0, patient.days_in_icu - 3)
        efficiency_bonus = days_past_recovery * 0.05

        return base_reward + efficiency_bonus

    # ---------------- OBSERVATION (Adding the Hint Mechanic) ----------------

    def _summary(self):
        summary_lines = []
        for w in self.wards:
            ward_info = f"Ward: {w.ward_name} ({w.total_beds} beds)"
            summary_lines.append(ward_info)
            for b in w.beds:
                features = []
                if b.has_ventilator: features.append("Ventilator")
                if b.is_negative_pressure: features.append("Negative Pressure")
                if b.has_specialized_mattress: features.append("Specialty Mattress")
                feature_str = ", ".join(features) if features else "Standard"
                
                status = "EMPTY"
                if b.current_occupant_id:
                    p = self.active_patients.get(b.current_occupant_id)
                    if p:
                        status = f"OCCUPIED by {p.patient_id} (GCS: {p.gcs_score}, Vent: {p.needs_ventilator}, Days in ICU: {p.days_in_icu})"
                    else:
                        status = f"OCCUPIED by {b.current_occupant_id}"
                        
                summary_lines.append(f"  - Bed {b.bed_id}: [{status}] Features: {feature_str}")
        return "\n".join(summary_lines)

    def _get_obs(self, reward, feedback="Environment Initialized."):
        
        # The Hint System
        hint = ""
        if self.fatal_errors >= self.HINT_AFTER_ERRORS and not self.done:
            hint = (
                "HINT: You are violating clinical constraints. Check if the patient "
                "needs a ventilator, isolation, or a specialty mattress. If all beds are full, "
                "you must use the STEP_DOWN action on a stable patient (GCS 14+, Days > 3) to free a bed."
            )

        # The Clamped Score
        if self.done and self.fatal_errors >= 3:
            final_score = 0.0
        else:
            final_score = _clamp(self.cumulative_reward)

        return ICUObservation(
            hospital_summary=self._summary(),
            unassigned_patients=self.unassigned_patients,
            feedback=feedback,
            hint=hint,
            reward=reward,
            done=self.done,
            score=final_score,
            step=self.time,
            fatal_errors=self.fatal_errors
        )

    @property
    def state(self):
        return {
            "unassigned_patients": self.unassigned_patients,
            "active_patients": self.active_patients,
            "wards": self.wards
        }