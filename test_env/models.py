from typing import List, Dict, Optional, Literal, Union, Any, Annotated
from pydantic import BaseModel, RootModel, Field

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
    total_beds: int
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
ICUAction = Annotated[Union[AssignBedAction, StepDownAction], Field(discriminator="action_type")]

class ICUActionRouter(BaseModel):
    """Adapter model so OpenEnv's HTTP/WebSocket server can validate actions."""
    
    @classmethod
    def model_validate(cls, data: Any, *args: Any, **kwargs: Any) -> ICUAction:  # type: ignore[override]
        action_type = (data or {}).get("action_type")

        if action_type == "ASSIGN_BED":
            return AssignBedAction(**data)
        if action_type == "STEP_DOWN":
            return StepDownAction(**data)

        raise ValueError(f"Unknown action_type: {action_type!r}")


# ==========================================
# OBSERVATION MODEL (What the Agent Sees)
# ==========================================

class ICUObservation(BaseModel):
    hospital_summary: str
    unassigned_patients: List[PatientState]
    feedback: str
    reward: float
    hint: str
    done: bool
    score: float
    step: int
    fatal_errors: int