# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ICU Bed Allocation environment client."""

from typing import Dict, Any
import sys
from pathlib import Path

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

# Ensure the repo root is in sys.path for direct script execution
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models import ICUAction, ICUObservation


# We use Dict[str, Any] for the State type because our new environment's 
# @property def state(self) returns a complex tracking dictionary.
class ICUEnv(EnvClient[ICUAction, ICUObservation, Dict[str, Any]]):

    # ---------------- STEP PAYLOAD ----------------

    def _step_payload(self, action: ICUAction) -> dict:
        # Convert the Pydantic Action model (AssignBedAction or StepDownAction)
        # directly into a dictionary payload for the environment server.
        return action.model_dump()

    # ---------------- PARSE RESULT ----------------

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {}) or {}
        # In the OpenEnv HTTP schema, reward/done
        # are surfaced at the top level of the response, not inside the
        # observation object.
        reward = payload.get("reward", obs_data.get("reward", 0.0))
        done = payload.get("done", obs_data.get("done", False))

        merged = {
            **obs_data,
            "reward": reward,
            "done": done,
        }

        observation = ICUObservation(**merged)

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    # ---------------- PARSE STATE ----------------

    def _parse_state(self, payload: dict) -> Dict[str, Any]:
        # The new environment state returns the unassigned_patients, 
        # active_patients registry, and ward capacity as a structured dict.
        return payload