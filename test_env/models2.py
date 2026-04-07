from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel

# ==========================================
# STATE MODELS
# ==========================================

class PatientState(BaseModel):
    patient_id: str
    age: int
    gcs_score: int
    needs_ventilator: bool
    is_infectious: bool
    has_severe_head_injury: bool
    has_paralysis: bool
    days_in_icu: int

class BedState(BaseModel):
    bed_id: str
    has_ventilator: bool
    is_negative_pressure: bool
    has_specialized_mattress: bool
    current_occupant_id: Optional[str] = None  

class WardState(BaseModel):
    ward_name: str
    beds: List[BedState]


# ==========================================
# ACTION MODELS (The Agent's Tools)
# ==========================================

class AssignBedAction(BaseModel):
    action_type: Literal["ASSIGN_BED"]
    patient_id: str
    bed_id: str

class StepDownAction(BaseModel):
    action_type: Literal["STEP_DOWN"]
    patient_id: str

# The Union type that OpenEnv uses to validate incoming actions
ICUAction = Union[AssignBedAction, StepDownAction]


# ==========================================
# OBSERVATION MODEL (What the Agent Sees)
# ==========================================

class ICUObservation(BaseModel):
    hospital_summary: str
    unassigned_patients: List[PatientState]
    feedback: str
    reward: float
    done: bool
    metadata: dict