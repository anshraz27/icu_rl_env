# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ICU Bed Allocation environment client."""

from typing import Dict
import sys
from pathlib import Path

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# Ensure the repo root is in sys.path for direct script execution
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from .models import ICUAction, ICUObservation, ICUState, Patient
except ImportError:
    from test_env.models import ICUAction, ICUObservation, ICUState


class ICUEnv(EnvClient[ICUAction, ICUObservation, ICUState]):

    # ---------------- STEP PAYLOAD ----------------

    def _step_payload(self, action: ICUAction) -> dict:
        return {
            "value": action.value
        }

    # ---------------- PARSE RESULT ----------------

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward", 0.0)

        # Parse patients list safely
        patients = [
            Patient(
                id=p.get("id", 0),
                severity=p.get("severity", 1),
                wait_time=p.get("wait_time", 0),
                survival_prob=p.get("survival_prob", 1.0),
            )
            for p in obs_data.get("patients", [])
        ]

        observation = ICUObservation(
            patients=patients,
            available_beds=obs_data.get("available_beds", 0),
            time_step=obs_data.get("time_step", 0),
            reward=reward,
            done=done,
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    # ---------------- PARSE STATE ----------------

    def _parse_state(self, payload: dict) -> ICUState:
        return ICUState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_patients=payload.get("total_patients", 0),
            total_admitted=payload.get("total_admitted", 0),
            deaths=payload.get("deaths", 0),
        )