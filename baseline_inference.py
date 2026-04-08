"""Baseline inference script using the OpenAI API client.

Runs an LLM agent (via OpenAI-compatible API) against all 3 ZeroWaste-Pattern
tasks and produces reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline_inference.py

    # Or with a custom model:
    python baseline_inference.py --model gpt-4o-mini --episodes 3

Requirements:
    pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from models.observation import Observation
from models.reward import Reward
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask


def get_openai_client():
    """Create an OpenAI client from environment variables.

    Returns:
        openai.OpenAI client instance.

    Raises:
        SystemExit: If OPENAI_API_KEY is not set or openai is not installed.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.")
        print("  pip install openai")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def build_system_prompt(task_name: str, fabric_w: float, fabric_l: float,
                        n_pieces: int, grid_w: int, grid_h: int,
                        max_rotations: int) -> str:
    """Build the system prompt describing the environment to the LLM.

    Args:
        task_name: Name of the current task.
        fabric_w: Fabric width in cm.
        fabric_l: Fabric max length in cm.
        n_pieces: Number of piece slots in action space.
        grid_w: Grid width (number of columns).
        grid_h: Grid height (number of rows).
        max_rotations: Maximum rotation index.

    Returns:
        System prompt string.
    """
    return f"""You are an AI agent playing the ZeroWaste-Pattern fabric cutting game.

TASK: {task_name}
FABRIC: {fabric_w}cm wide x {fabric_l}cm long
GRID: {grid_w} columns x {grid_h} rows (each cell = 2cm)

You must place garment pattern pieces on fabric to MAXIMIZE utilization and MINIMIZE waste.

ACTION FORMAT: You must respond with ONLY a JSON object (no other text):
{{"piece_idx": <int 0-{n_pieces-1}>, "grid_x": <int 0-{grid_w-1}>, "grid_y": <int 0-{grid_h-1}>, "rotation_idx": <int 0-{max_rotations-1}>}}

RULES:
- piece_idx selects which remaining piece to place (lower = larger pieces usually)
- grid_x, grid_y set the placement position on the fabric grid
- rotation_idx: 0=0°, 1=90°, 2=180°, 3=270°
- Pieces must not overlap or go out of bounds
- Place pieces close together at the bottom-left to minimize waste
- Larger pieces should be placed first

STRATEGY: Pack pieces tightly from the bottom-left corner. Try piece_idx=0 first (largest remaining), place near y=0, sweep x from 0 upward."""


def build_observation_prompt(obs: Observation, step: int,
                             last_reward: float, last_valid: bool,
                             pieces_placed: int, pieces_total: int) -> str:
    """Build a concise observation prompt for the LLM.

    Args:
        obs: Typed Observation from the environment.
        step: Current step number.
        last_reward: Reward from the last step.
        last_valid: Whether the last action was valid.
        pieces_placed: Number of pieces placed so far.
        pieces_total: Total number of pieces to place.

    Returns:
        User message string describing current state.
    """
    # Summarize remaining pieces
    remaining = []
    for i, row in enumerate(obs.pieces_remaining):
        if row[7] > 0.5:  # is_present flag
            remaining.append(
                f"  piece {i}: w={row[0]:.2f} h={row[1]:.2f} area={row[2]:.4f}"
            )

    remaining_str = "\n".join(remaining[:10])  # limit to first 10
    if len(remaining) > 10:
        remaining_str += f"\n  ... and {len(remaining) - 10} more"

    return f"""Step {step} | Util: {obs.utilization:.1%} | Placed: {pieces_placed}/{pieces_total} | Last: {"VALID" if last_valid else "INVALID"} (r={last_reward:.3f})

Remaining pieces (index: normalized_width, height, area_ratio):
{remaining_str}

Fabric used: {obs.fabric_length_used:.1%} of max length.

Choose your next action as JSON:"""


def parse_llm_action(response_text: str, n_pieces: int, grid_w: int,
                     grid_h: int, max_rotations: int) -> Optional[np.ndarray]:
    """Parse the LLM's response into a valid action array.

    Args:
        response_text: Raw text from the LLM.
        n_pieces: Max piece index.
        grid_w: Grid width.
        grid_h: Grid height.
        max_rotations: Max rotation index.

    Returns:
        Numpy action array of shape (4,), or None if parsing fails.
    """
    text = response_text.strip()

    # Try to extract JSON from the response
    import re
    json_match = re.search(r'\{[^}]+\}', text)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        piece_idx = int(data.get("piece_idx", 0))
        grid_x = int(data.get("grid_x", 0))
        grid_y = int(data.get("grid_y", 0))
        rotation_idx = int(data.get("rotation_idx", 0))
    except (ValueError, TypeError):
        return None

    # Clamp to valid ranges
    piece_idx = max(0, min(piece_idx, n_pieces))
    grid_x = max(0, min(grid_x, grid_w - 1))
    grid_y = max(0, min(grid_y, grid_h - 1))
    rotation_idx = max(0, min(rotation_idx, max_rotations - 1))

    return np.array([piece_idx, grid_x, grid_y, rotation_idx], dtype=np.int64)


def run_llm_episode(
    client,
    model: str,
    env: ZeroWasteFabricEnv,
    task_name: str,
    max_steps: int = 50,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single episode using the LLM as the agent.

    Args:
        client: OpenAI client instance.
        model: Model name (e.g., "gpt-4o-mini").
        env: The ZeroWaste environment.
        task_name: Name for display.
        max_steps: Max LLM calls per episode.
        seed: Random seed for reproducibility.
        verbose: Print step details.

    Returns:
        Dictionary with episode metrics.
    """
    obs_raw, info = env.reset(seed=seed)
    obs = Observation.from_gym_obs(obs_raw)

    system_prompt = build_system_prompt(
        task_name=task_name,
        fabric_w=env.fabric_width,
        fabric_l=env.fabric_max_length,
        n_pieces=env.max_pieces,
        grid_w=env.grid_w,
        grid_h=env.grid_h,
        max_rotations=4,
    )

    messages = [{"role": "system", "content": system_prompt}]
    total_reward = 0.0
    last_reward = 0.0
    last_valid = True
    step = 0
    parse_failures = 0

    for step_num in range(1, max_steps + 1):
        step = step_num

        user_msg = build_observation_prompt(
            obs=obs,
            step=step_num,
            last_reward=last_reward,
            last_valid=last_valid,
            pieces_placed=info.get("pieces_placed", 0),
            pieces_total=info.get("pieces_total", 0),
        )

        messages.append({"role": "user", "content": user_msg})

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.1,  # low temperature for reproducibility
            )
            assistant_msg = response.choices[0].message.content or ""
        except Exception as e:
            if verbose:
                print(f"  LLM error at step {step_num}: {e}")
            assistant_msg = '{"piece_idx": 0, "grid_x": 0, "grid_y": 0, "rotation_idx": 0}'

        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse action
        action = parse_llm_action(
            assistant_msg, env.max_pieces, env.grid_w, env.grid_h, 4
        )

        if action is None:
            parse_failures += 1
            if verbose:
                print(f"  Step {step_num}: parse failure, using fallback")
            action = env.action_space.sample()

        # Execute step
        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs = Observation.from_gym_obs(obs_raw)
        reward_obj = Reward.from_step_info(reward, info)

        total_reward += reward
        last_reward = reward
        last_valid = reward_obj.is_valid

        if verbose:
            print(
                f"  Step {step_num}: action={action.tolist()} "
                f"valid={last_valid} reward={reward:.3f} "
                f"util={obs.utilization:.1%}"
            )

        if terminated or truncated:
            break

        # Keep message history manageable (sliding window)
        if len(messages) > 20:
            messages = [messages[0]] + messages[-10:]

    # Get final state and grade
    final_state = env.state()
    grade = env.task.grade(final_state)

    return {
        "task": task_name,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "utilization_pct": round(final_state.utilization_pct, 2),
        "pieces_placed": final_state.n_placed,
        "pieces_total": final_state.n_placed + final_state.n_remaining,
        "invalid_actions": final_state.invalid_action_count,
        "grade": round(grade, 4),
        "parse_failures": parse_failures,
    }


def run_random_baseline(
    env: ZeroWasteFabricEnv,
    task_name: str,
    n_episodes: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run random agent baseline for comparison.

    Args:
        env: Environment instance.
        task_name: Task name for display.
        n_episodes: Number of episodes to average.
        seed: Starting seed.

    Returns:
        Averaged metrics dictionary.
    """
    grades = []
    utils = []
    placed = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc

        state = env.state()
        grade = env.task.grade(state)
        grades.append(grade)
        utils.append(state.utilization_pct)
        placed.append(state.n_placed)

    return {
        "task": task_name,
        "avg_grade": round(float(np.mean(grades)), 4),
        "avg_utilization_pct": round(float(np.mean(utils)), 2),
        "avg_pieces_placed": round(float(np.mean(placed)), 1),
    }


def main() -> None:
    """Run baseline inference on all 3 tasks."""
    parser = argparse.ArgumentParser(
        description="ZeroWaste-Pattern baseline inference using OpenAI API"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per task for LLM agent (default: 1)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max LLM calls per episode (default: 30)"
    )
    parser.add_argument(
        "--random-episodes", type=int, default=5,
        help="Episodes for random baseline (default: 5)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print step-by-step details"
    )
    parser.add_argument(
        "--random-only", action="store_true",
        help="Only run random baseline (no API key needed)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ZeroWaste-Pattern: Baseline Inference")
    print("=" * 70)

    tasks = [
        ("BasicPacking", BasicPackingTask()),
        ("IrregularShapes", IrregularShapesTask()),
        ("IndustrialMode", IndustrialModeTask()),
    ]

    # --- Random baseline (always runs) ---
    print("\n--- Random Agent Baseline ---")
    random_results = []
    for name, task in tasks:
        env = ZeroWasteFabricEnv(task=task, max_steps=200)
        result = run_random_baseline(env, name, n_episodes=args.random_episodes)
        random_results.append(result)
        print(
            f"  {name:20s}  grade={result['avg_grade']:.4f}  "
            f"util={result['avg_utilization_pct']:5.1f}%  "
            f"placed={result['avg_pieces_placed']:.1f}"
        )

    # --- LLM agent ---
    if not args.random_only:
        client = get_openai_client()
        print(f"\n--- LLM Agent Baseline (model={args.model}) ---")
        llm_results = []
        for name, task in tasks:
            env = ZeroWasteFabricEnv(task=task, max_steps=200)
            for ep in range(args.episodes):
                t0 = time.time()
                result = run_llm_episode(
                    client=client,
                    model=args.model,
                    env=env,
                    task_name=name,
                    max_steps=args.max_steps,
                    seed=ep,
                    verbose=args.verbose,
                )
                dt = time.time() - t0
                llm_results.append(result)
                print(
                    f"  {name:20s} ep={ep}  grade={result['grade']:.4f}  "
                    f"util={result['utilization_pct']:5.1f}%  "
                    f"placed={result['pieces_placed']}/{result['pieces_total']}  "
                    f"invalid={result['invalid_actions']}  "
                    f"t={dt:.1f}s"
                )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Baseline Scores (grade 0.0-1.0)")
    print("=" * 70)
    print(f"{'Task':<22} {'Random':>10} ", end="")
    if not args.random_only:
        print(f"{'LLM':>10}", end="")
    print()
    print("-" * 50)

    for i, (name, _) in enumerate(tasks):
        rg = random_results[i]["avg_grade"]
        print(f"{name:<22} {rg:>10.4f} ", end="")
        if not args.random_only:
            task_llm = [r for r in llm_results if r["task"] == name]
            if task_llm:
                avg_grade = np.mean([r["grade"] for r in task_llm])
                print(f"{avg_grade:>10.4f}", end="")
        print()

    # Save results to JSON
    output = {
        "random_baseline": random_results,
    }
    if not args.random_only:
        output["llm_baseline"] = llm_results
        output["model"] = args.model

    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to baseline_scores.json")


if __name__ == "__main__":
    main()
