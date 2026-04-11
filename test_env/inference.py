"""
Inference Script for ICU Triage Environment (Client-Server Mode)
================================================================
Runs the LLM agent against the OpenEnv server using the custom ICUEnv client.
"""

import os
import sys
import json
import textwrap
import argparse
from openai import OpenAI

# Ensure current directory is in sys.path for local/remote compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import your custom client and models

from client import ICUEnv
from models import AssignBedAction, StepDownAction

from dotenv import load_dotenv

# Load environment variables from a .env file (project root)
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
MAX_STEPS = 15
TEMPERATURE = 0.0 # Deterministic scores for reproducibility

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the Chief Medical Triage AI for a hospital ICU.
    Your goal is to clear the 'Unassigned Patients' queue by assigning them to appropriate beds.
    
    CLINICAL RULES:
    1. Minors (age < 18) must go to the PICU.
    2. Infectious patients MUST go to an Isolation ICU bed (negative pressure).
    3. Head injury patients MUST go to the Neuro ICU.
    4. Trauma patients MUST go to the Trauma ICU.
    5. Ventilator-dependent patients MUST be assigned a bed with a ventilator.
    6. Paralyzed patients MUST be assigned a bed with a specialty mattress.
    
    CAPACITY RULES:
    If no beds are available that meet a critical patient's needs, you MUST use the STEP_DOWN action 
    on a currently admitted patient to free up their bed. 
    You can ONLY step down a patient if: GCS is 14+, they do not need a ventilator, and Days in ICU >= 3.
    
    OUTPUT FORMAT:
    You must output ONLY a valid JSON object representing your action. Do not include markdown formatting.
    
    Format for Assigning a Bed:
    {"action_type": "ASSIGN_BED", "patient_id": "PT-01", "bed_id": "S1"}
    
    Format for Stepping Down a stable patient to free a bed:
    {"action_type": "STEP_DOWN", "patient_id": "PT-04"}
    """
).strip()


def build_user_prompt(step: int, observation, full_state) -> str:
    """Formats the environment observation into a clear text prompt for the LLM."""
    
    unassigned_str = "\n".join(
        f"- ID: {p.patient_id} | Age: {p.age} | GCS: {p.gcs_score} | Vent: {p.needs_ventilator} | "
        f"Infectious: {p.is_infectious} | Head Trauma: {p.has_severe_head_injury} | Paralysis: {p.has_paralysis}"
        for p in observation.unassigned_patients
    ) if observation.unassigned_patients else "None"

    # full_state is a dict returned by client2._parse_state, so
    # active_patients entries are plain dicts, not Pydantic models.
    active_patients = {}
    if isinstance(full_state, dict):
        active_patients = full_state.get("active_patients") or {}

    active_str = "\n".join(
        f"- ID: {p.get('patient_id')} | GCS: {p.get('gcs_score')} | "
        f"Vent: {p.get('needs_ventilator')} | Days in ICU: {p.get('days_in_icu')}"
        for p in active_patients.values()
    ) if active_patients else "None"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        
        --- HOSPITAL STATUS ---
        {observation.hospital_summary}
        
        --- UNASSIGNED PATIENTS (WAITING ROOM) ---
        {unassigned_str}
        
        --- ACTIVE PATIENTS IN BEDS (CANDIDATES FOR STEP-DOWN) ---
        {active_str}
        
        --- PREVIOUS ACTION FEEDBACK ---
        {observation.feedback}
        
        --- SYSTEM HINT ---
        {getattr(observation, 'hint', '')}
        
        Based on the current state, output your next JSON action.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str):
    """Safely parses the LLM's JSON string into a Pydantic Action Model."""
    try:
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        action_dict = json.loads(clean_text)
        
        if action_dict.get("action_type") == "ASSIGN_BED":
            return AssignBedAction(**action_dict)
        elif action_dict.get("action_type") == "STEP_DOWN":
            return StepDownAction(**action_dict)
        else:
            raise ValueError("Unknown action_type in JSON.")
            
    except Exception as e:
        print(f"Failed to parse LLM output: {response_text}. Error: {e}")
        return StepDownAction(action_type="STEP_DOWN", patient_id="PARSE_ERROR")


def run_episode(client: OpenAI, env: ICUEnv, episode_num: int) -> float:
    """Runs a single episode via the EnvClient and returns the final score."""
    print(f"\n{'='*40}")
    print(f"Starting Episode {episode_num}")
    print(f"{'='*40}")
    
    # reset() on the client returns a StepResult wrapper
    result = env.reset()
    obs = result.observation
    
    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        # Note: env.state() is now a method call on the client
        current_state = env.state()
        user_prompt = build_user_prompt(step, obs, current_state)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"} 
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"API Request failed: {exc}")
            # End the episode gracefully if the LLM call fails
            break

        action_model = parse_model_action(response_text)
        print(f"Step {step} LLM Action: {action_model.model_dump()}")
        
        # step() on the client returns a StepResult wrapper
        result = env.step(action_model)
        obs = result.observation
        
        print(f"Reward: {result.reward:+.2f} | Done: {result.done} | Feedback: {obs.feedback}")

    final_score = obs.score
    print(f"Episode {episode_num} Complete. Final Score: {final_score}")
    return final_score


def main() -> None:
    # 1. Parse URL from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001", help="URL of the OpenEnv server")
    args = parser.parse_args()

    # 2. Ensure credentials are set
    if not API_KEY:
        print("Error: API Key not found. Please set HF_TOKEN or GROQ_API_KEY.")
        return
        
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    baseline_scores = []

    # 3. Initialize the Custom Environment Client in synchronous mode.
    # .sync() wraps the async EnvClient methods so reset/step/state are
    # regular blocking calls returning StepResult objects.
    with ICUEnv(base_url=args.url).sync() as env:
        # Run 3 consecutive episodes.
        # (reset() in env2 automatically cycles easy -> medium -> hard)
        for i in range(1, 4):
            score = run_episode(llm_client, env, i)
            baseline_scores.append(score)

    print("\n\n" + "="*40)
    print("OPENENV BASELINE EVALUATION RESULTS")
    print("="*40)
    for i, score in enumerate(baseline_scores, 1):
        print(f"Episode {i}: {score:.2f} / 1.00")
    
    if baseline_scores:
        average_score = sum(baseline_scores) / len(baseline_scores)
        print(f"\nOverall Baseline Average: {average_score:.2f}")


if __name__ == "__main__":
    main()