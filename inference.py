"""OpenEnv inference script for ZeroWaste-Pattern.

Runs an LLM agent via the provided LiteLLM proxy against all 3 tasks.
Produces structured [START]/[STEP]/[END] output on stdout.

Uses environment variables:
    API_BASE_URL  - LiteLLM proxy base URL (injected by validator)
    API_KEY       - API key for the proxy (injected by validator)
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from env.fabric_env import ZeroWasteFabricEnv
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask


TASKS = [
    ("BasicPacking", BasicPackingTask),
    ("IrregularShapes", IrregularShapesTask),
    ("IndustrialMode", IndustrialModeTask),
]


def log(msg: str) -> None:
    """Print to stdout with flush."""
    print(msg, flush=True)


def get_client() -> OpenAI:
    """Create OpenAI client using the injected proxy env vars."""
    base_url = os.environ.get("API_BASE_URL", os.environ.get("OPENAI_API_BASE", ""))
    api_key = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    if not base_url or not api_key:
        log("[WARN] API_BASE_URL or API_KEY not set, falling back to defaults")

    return OpenAI(
        base_url=base_url if base_url else None,
        api_key=api_key if api_key else "dummy",
    )


def build_system_prompt(task_name: str, env: ZeroWasteFabricEnv) -> str:
    """Build system prompt for the LLM agent."""
    return (
        f"You are an AI agent placing garment pattern pieces on fabric to minimize waste.\n"
        f"Task: {task_name}. Fabric: {env.fabric_width:.0f}cm wide x {env.fabric_max_length:.0f}cm long.\n"
        f"Grid: {env.grid_w} cols x {env.grid_h} rows (each cell = 2cm).\n\n"
        f"You must respond with ONLY a JSON object, no other text:\n"
        f'  {{"piece_idx": <int 0-{env.max_pieces-1}>, "grid_x": <int 0-{env.grid_w-1}>, '
        f'"grid_y": <int 0-{env.grid_h-1}>, "rotation_idx": <int 0-3>}}\n\n'
        f"STRATEGY:\n"
        f"- piece_idx: choose the lowest available index (largest pieces first)\n"
        f"- grid_x, grid_y: pack pieces tightly from bottom-left (low y, low x)\n"
        f"- rotation_idx: 0=0deg, 1=90deg, 2=180deg, 3=270deg\n"
        f"- Avoid overlaps by incrementing grid_y when lower rows are full"
    )


def parse_action(text: str) -> dict | None:
    """Extract JSON action from LLM response."""
    match = re.search(r'\{[^}]+\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_task(client: OpenAI, model: str, task_name: str, TaskCls,
             seed: int = 42, max_llm_steps: int = 40) -> None:
    """Run a single task with the LLM agent and emit structured output."""
    task = TaskCls()
    env = ZeroWasteFabricEnv(task=task, max_steps=200)
    obs, info = env.reset(seed=seed)

    log(f"[START] task={task_name}")

    system_prompt = build_system_prompt(task_name, env)
    messages = [{"role": "system", "content": system_prompt}]

    total_reward = 0.0
    step_num = 0
    done = False
    last_valid = True

    while not done:
        step_num += 1

        # Use LLM for first max_llm_steps, then fall back to random
        if step_num <= max_llm_steps:
            # Build user message
            remaining = info.get("pieces_remaining", 0)
            util_pct = info.get("utilization_pct", 0.0)
            placed = info.get("pieces_placed", 0)
            total = info.get("pieces_total", 0)

            user_msg = (
                f"Step {step_num} | Util: {util_pct:.1f}% | "
                f"Placed: {placed}/{total} | "
                f"Last: {'VALID' if last_valid else 'INVALID'}\n"
                f"Pick next action as JSON:"
            )
            messages.append({"role": "user", "content": user_msg})

            # Call the LLM through the proxy
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=80,
                    temperature=0.2,
                )
                reply = response.choices[0].message.content or ""
            except Exception as e:
                reply = '{"piece_idx": 0, "grid_x": 0, "grid_y": 0, "rotation_idx": 0}'

            messages.append({"role": "assistant", "content": reply})

            # Parse action from LLM
            parsed = parse_action(reply)
            if parsed:
                action = np.array([
                    int(parsed.get("piece_idx", 0)),
                    int(parsed.get("grid_x", 0)),
                    int(parsed.get("grid_y", 0)),
                    int(parsed.get("rotation_idx", 0)),
                ], dtype=np.int64)
            else:
                action = env.action_space.sample()

            # Trim message history to stay within context limits
            if len(messages) > 20:
                messages = [messages[0]] + messages[-10:]
        else:
            # Random fallback for remaining steps
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        last_valid = info.get("action_result", {}).get("valid", True)

        log(f"[STEP] step={step_num} reward={reward:.4f}")

    # Final grading
    state = env.state()
    grade = env.task.grade(state)

    log(
        f"[END] task={task_name} "
        f"score={grade:.4f} "
        f"steps={step_num} "
        f"utilization={state.utilization_pct:.2f} "
        f"pieces_placed={state.n_placed} "
        f"total_reward={total_reward:.4f}"
    )


def main() -> None:
    client = get_client()

    # Try to detect model from env, default to gpt-4o-mini
    model = os.environ.get("MODEL", "gpt-4o-mini")
    seed = 42

    for task_name, TaskCls in TASKS:
        run_task(client, model, task_name, TaskCls, seed=seed)


if __name__ == "__main__":
    main()
