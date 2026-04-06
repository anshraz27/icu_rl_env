import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from test_env.client import ICUEnv
from test_env.models import ICUAction


# ---------------- SYSTEM PROMPT ----------------

SYSTEM_PROMPT = """You are an ICU triage decision-making agent.

Your goal is to maximize patient survival and minimize deaths.

You must choose ONE action at every step:
- 0 → wait (do nothing)
- 1..k → admit the patient at index (1-based)

CRITICAL RULES (STRICT):

1. NEVER leave critical patients (severity 4 or 5) waiting if a bed is available.
2. Waiting is dangerous — patients lose survival probability every step.
3. If beds are available AND patients exist → you SHOULD admit someone.
4. Always prioritize patients with higher severity.
5. Avoid admitting low severity patients (1–2) if higher severity patients are waiting.
6. Long waiting queues lead to penalties — act quickly.
7. Doing nothing (action=0) is only acceptable if:
   - there are NO patients, OR
   - there are NO available beds.

STRATEGY:
- Prefer severity 5 > 4 > 3 > 2 > 1
- Admit the most critical patient first
- Do not delay decisions

OUTPUT FORMAT (STRICT):
- Return ONLY a single integer
- No explanation, no text
- Example outputs: 0, 1, 2, 3

Your decision directly impacts survival — act decisively.
"""


# ---------------- BUILD PROMPT ----------------

def build_prompt(obs):

    patient_desc = []
    for i, p in enumerate(obs.patients):
        patient_desc.append(
            f"{i+1}: severity={p.severity}, wait={p.wait_time}"
        )

    parts = [
    f"Time step: {obs.time_step}",
    f"Available beds: {obs.available_beds}",
    f"Patients waiting: {len(obs.patients)}",
    "",
    "Patients (index: severity, wait_time):",
    "\n".join(patient_desc) if patient_desc else "No patients",
   ]

    if obs.metadata.get("feedback"):
        parts += ["", f"Feedback: {obs.metadata['feedback']}"]

    if obs.metadata.get("hint"):
        parts += ["", f"Hint: {obs.metadata['hint']}"]

    parts += ["", "Choose action (0 = wait, 1..k = admit):"]

    return "\n".join(parts)


# ---------------- RUN EPISODE ----------------

def run_episode(env, client, verbose=True):

    result = env.reset()
    obs = result.observation

    total_reward = 0.0

    while not result.done:
        prompt = build_prompt(obs)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )

        try:
            action_val = int(response.choices[0].message.content.strip())
        except:
            action_val = 0  # fallback

        if verbose:
            print(f"Action chosen: {action_val}")

        result = env.step(ICUAction(value=action_val))
        obs = result.observation

        total_reward += result.reward or 0.0

        if verbose:
            reward_val = result.reward or 0.0
            score_val = obs.metadata.get('score')
            if score_val is None:
                score_val = reward_val  # Fallback to reward, as upd_env.py sets reward = score
            
            print(
                f"Reward: {reward_val:.3f} | "
                f"Score: {score_val:.3f}"
            )

    final_score = obs.metadata.get("score")
    if final_score is None:
        final_score = result.reward or 0.0

    print(f"\nFinal Score: {final_score:.3f}")
    return final_score


# ---------------- MAIN ----------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY first")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    print("\nICU Environment - LLM Agent\n")

    with ICUEnv(base_url=args.url).sync() as env:
        run_episode(env, client)


if __name__ == "__main__":
    main()