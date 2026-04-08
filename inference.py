"""OpenEnv inference script for ZeroWaste-Pattern.

Runs a model (via OpenAI API) against all 3 tasks and reports
reproducible baseline scores (grade 0.0-1.0).

Usage:
    export OPENAI_API_KEY=sk-...
    python inference.py

    # Random-only (no API key needed):
    python inference.py --random-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from models.observation import Observation
from models.reward import Reward
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask


# ------------------------------------------------------------------
# Tasks registry
# ------------------------------------------------------------------

TASKS = [
    ("BasicPacking", BasicPackingTask, "easy"),
    ("IrregularShapes", IrregularShapesTask, "medium"),
    ("IndustrialMode", IndustrialModeTask, "hard"),
]


# ------------------------------------------------------------------
# Random baseline agent
# ------------------------------------------------------------------

def run_random_agent(env: ZeroWasteFabricEnv, task_name: str,
                     n_episodes: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Run random agent and return averaged metrics."""
    grades, utils, placed_counts = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

        state = env.state()
        grade = env.task.grade(state)
        grades.append(grade)
        utils.append(state.utilization_pct)
        placed_counts.append(state.n_placed)

    return {
        "task": task_name,
        "agent": "random",
        "avg_grade": round(float(np.mean(grades)), 4),
        "std_grade": round(float(np.std(grades)), 4),
        "avg_utilization_pct": round(float(np.mean(utils)), 2),
        "avg_pieces_placed": round(float(np.mean(placed_counts)), 1),
        "episodes": n_episodes,
    }


# ------------------------------------------------------------------
# LLM agent (OpenAI API)
# ------------------------------------------------------------------

def run_llm_agent(env: ZeroWasteFabricEnv, task_name: str,
                  model: str = "gpt-4o-mini", seed: int = 0,
                  max_steps: int = 30) -> Dict[str, Any]:
    """Run one episode using an LLM via the OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    obs_raw, info = env.reset(seed=seed)
    obs = Observation.from_gym_obs(obs_raw)

    system_prompt = (
        f"You are an AI agent placing garment pieces on fabric to minimize waste.\n"
        f"Task: {task_name}. Fabric: {env.fabric_width}x{env.fabric_max_length}cm.\n"
        f"Grid: {env.grid_w}cols x {env.grid_h}rows. Cell=2cm.\n\n"
        f"Reply ONLY with JSON: {{\"piece_idx\": int, \"grid_x\": int, \"grid_y\": int, \"rotation_idx\": int}}\n"
        f"Ranges: piece_idx 0-{env.max_pieces-1}, grid_x 0-{env.grid_w-1}, "
        f"grid_y 0-{env.grid_h-1}, rotation_idx 0-3.\n"
        f"Pack pieces tightly from bottom-left. Place largest pieces first."
    )

    messages = [{"role": "system", "content": system_prompt}]
    total_reward = 0.0
    last_valid = True
    steps = 0

    for step_num in range(1, max_steps + 1):
        steps = step_num

        # Build compact user message
        remaining = sum(1 for row in obs.pieces_remaining if row[7] > 0.5)
        user_msg = (
            f"Step {step_num} | Util: {obs.utilization:.1%} | "
            f"Remaining: {remaining} pieces | "
            f"Last: {'OK' if last_valid else 'INVALID'}\n"
            f"Choose action:"
        )
        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=80, temperature=0.1,
            )
            reply = response.choices[0].message.content or ""
        except Exception:
            reply = '{"piece_idx":0,"grid_x":0,"grid_y":0,"rotation_idx":0}'

        messages.append({"role": "assistant", "content": reply})

        # Parse action
        import re
        action = None
        match = re.search(r'\{[^}]+\}', reply)
        if match:
            try:
                d = json.loads(match.group())
                action = np.array([
                    int(d.get("piece_idx", 0)),
                    int(d.get("grid_x", 0)),
                    int(d.get("grid_y", 0)),
                    int(d.get("rotation_idx", 0)),
                ], dtype=np.int64)
            except (json.JSONDecodeError, ValueError):
                pass

        if action is None:
            action = env.action_space.sample()

        obs_raw, reward, term, trunc, info = env.step(action)
        obs = Observation.from_gym_obs(obs_raw)
        total_reward += reward
        last_valid = info.get("action_result", {}).get("valid", True)

        if term or trunc:
            break

        # Keep messages short
        if len(messages) > 16:
            messages = [messages[0]] + messages[-8:]

    state = env.state()
    grade = env.task.grade(state)

    return {
        "task": task_name,
        "agent": f"llm:{model}",
        "grade": round(grade, 4),
        "utilization_pct": round(state.utilization_pct, 2),
        "pieces_placed": state.n_placed,
        "pieces_total": state.n_placed + state.n_remaining,
        "steps": steps,
        "total_reward": round(total_reward, 4),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZeroWaste-Pattern inference")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--episodes", type=int, default=5, help="Random agent episodes")
    parser.add_argument("--random-only", action="store_true", help="Skip LLM agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("ZeroWaste-Pattern: Inference")
    print("=" * 60)

    results = {"random": [], "llm": []}

    # Random baseline
    print("\n--- Random Agent ---")
    for name, TaskCls, difficulty in TASKS:
        env = ZeroWasteFabricEnv(task=TaskCls(), max_steps=200)
        r = run_random_agent(env, name, n_episodes=args.episodes, seed=args.seed)
        results["random"].append(r)
        print(f"  [{difficulty:6s}] {name:20s}  grade={r['avg_grade']:.4f}  util={r['avg_utilization_pct']:5.1f}%")

    # LLM agent
    if not args.random_only:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\n[SKIP] OPENAI_API_KEY not set. Use --random-only or set the key.")
        else:
            print(f"\n--- LLM Agent ({args.model}) ---")
            for name, TaskCls, difficulty in TASKS:
                env = ZeroWasteFabricEnv(task=TaskCls(), max_steps=200)
                t0 = time.time()
                r = run_llm_agent(env, name, model=args.model, seed=args.seed)
                dt = time.time() - t0
                results["llm"].append(r)
                print(
                    f"  [{difficulty:6s}] {name:20s}  grade={r['grade']:.4f}  "
                    f"util={r['utilization_pct']:5.1f}%  "
                    f"placed={r['pieces_placed']}/{r['pieces_total']}  "
                    f"t={dt:.1f}s"
                )

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Task':<22} {'Difficulty':<10} {'Random':>10}", end="")
    if results["llm"]:
        print(f" {'LLM':>10}", end="")
    print()
    print("-" * 55)
    for i, (name, _, diff) in enumerate(TASKS):
        rg = results["random"][i]["avg_grade"]
        print(f"{name:<22} {diff:<10} {rg:>10.4f}", end="")
        if results["llm"]:
            lg = results["llm"][i]["grade"]
            print(f" {lg:>10.4f}", end="")
        print()

    # Save
    with open("inference_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to inference_scores.json")


if __name__ == "__main__":
    main()
