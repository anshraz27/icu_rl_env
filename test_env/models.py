from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# --------------- action -----------

class ICUAction(Action, BaseModel):
    """Action for the ICU environment.

    0  -> wait
    1+ -> admit patient at index (value - 1)
    """

    value: int = Field(..., ge=0)


# --------------- patient -----------

class Patient(BaseModel):
    id: int
    severity: int = Field(..., ge=1, le=5)
    wait_time: int = Field(..., ge=0)
    survival_prob: float = Field(..., ge=0.0, le=1.0)


# --------------- observation -----------

class ICUObservation(Observation, BaseModel):
    patients: List[Patient] = Field(default_factory=list)
    available_beds: int = Field(..., ge=0)
    time_step: int = Field(..., ge=0)

    reward: float
    done: bool

    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------- state -----------

class ICUState(State, BaseModel):
    episode_id: Optional[str]
    step_count: int = Field(..., ge=0)

    total_patients: int = Field(..., ge=0)
    total_admitted: int = Field(..., ge=0)
    deaths: int = Field(..., ge=0)