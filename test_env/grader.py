"""
Grader/evaluator for the ICU Bed Allocation environment (FINAL).

Aligned with SQL-style graded ICU environment:
- Uses metadata["score"] (primary signal)
- Tracks reward + score
- Uses env.state property correctly
- Handles feedback + hint (debuggable)
"""

from dataclasses import dataclass
from typing import Protocol, Tuple, Dict, List
import sys
from pathlib import Path
import random

# ---------------- SETUP ----------------

random.seed(42)

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from test_env.client import ICUEnv
from test_env.models import ICUAction, ICUObservation


# ---------------- POLICY INTERFACE ----------------

class Policy(Protocol):
    def act(self, obs: ICUObservation) -> ICUAction:
        ...


# ---------------- BASELINES ----------------

@dataclass
class RandomICUPolicy:
    def act(self, obs: ICUObservation) -> ICUAction:
        if obs.patients and obs.available_beds > 0:
            idx = random.randrange(len(obs.patients))
            return ICUAction(value=idx + 1)
        return ICUAction(value=0)


@dataclass
class GreedyICUPolicy:
    def act(self, obs: ICUObservation) -> ICUAction:
        if obs.available_beds > 0 and obs.patients:
            idx = max(range(len(obs.patients)), key=lambda i: obs.patients[i].severity)
            return ICUAction(value=idx + 1)
        return ICUAction(value=0)


# ---------------- EPISODE ----------------

def run_episode(
    env: ICUEnv,
    policy: Policy,
    max_steps: int = 200
) -> Tuple[float, float, int, int]:

    result = env.reset()
    obs = result.observation

    task_level = obs.metadata.get("task_level", 1)

    total_reward = 0.0
    steps = 0

    while not result.done and steps < max_steps:
        action = policy.act(obs)

        # Debug invalid actions
        if action.value > len(obs.patients):
            print("Invalid action:", action.value)

        result = env.step(action)
        obs = result.observation

        total_reward += result.reward or 0.0
        steps += 1

    # FINAL SCORE (always from metadata now)
    score = obs.metadata.get("score", 0.0)

    # Fallback (rare)
    if score is None:
        state = env.state  
        if state.total_patients > 0:
            score = max(0.0, 1.0 - (state.deaths / state.total_patients))
        else:
            score = 1.0

    return score, total_reward, steps, task_level


# ---------------- EVALUATION ----------------

def evaluate(policy: Policy, episodes: int = 15) -> None:

    total_steps = 0

    task_scores: Dict[int, List[float]] = {1: [], 2: [], 3: []}
    task_rewards: Dict[int, List[float]] = {1: [], 2: [], 3: []}

    with ICUEnv(base_url="http://localhost:8000").sync() as env:
        for _ in range(episodes):
            score, reward, steps, level = run_episode(env, policy)

            if level in task_scores:
                task_scores[level].append(score)
                task_rewards[level].append(reward)

            total_steps += steps

    print("\nICU Environment Evaluation Results")
    print("=====================================")
    print(f"Total Episodes: {episodes}\n")

    for level in [1, 2, 3]:
        scores = task_scores[level]
        rewards = task_rewards[level]

        if scores:
            avg_score = sum(scores) / len(scores)
            avg_reward = sum(rewards) / len(rewards)

            print(
                f"Task {level} | Episodes: {len(scores):02d} | "
                f"Avg Score: {avg_score:.3f} | "
                f"Avg Reward: {avg_reward:.3f}"
            )
        else:
            print(f"Task {level} | No episodes")

    avg_steps = total_steps / episodes if episodes else 0.0

    print("\n-------------------------------------")
    print(f"Avg Episode Length: {avg_steps:.1f} steps")
    print("=====================================\n")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    print("nsure ICU server is running at http://localhost:8000\n")

    print("Running Random Policy...")
    evaluate(RandomICUPolicy(), episodes=15)

    print("\nRunning Greedy Policy...")
    evaluate(GreedyICUPolicy(), episodes=15)