"""OpenEnv inference script for ZeroWaste-Pattern.

Produces structured [START]/[STEP]/[END] output on stdout as required
by the OpenEnv validator.

Usage:
    python inference.py
    OPENAI_API_KEY=sk-... python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def run_task(task_name: str, TaskCls, seed: int = 42) -> None:
    """Run a single task and emit structured output."""
    task = TaskCls()
    env = ZeroWasteFabricEnv(task=task, max_steps=200)
    obs, info = env.reset(seed=seed)

    log(f"[START] task={task_name}")

    total_reward = 0.0
    step_num = 0
    done = False

    while not done:
        step_num += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

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
    seed = 42

    # If OPENAI_API_KEY is set, we could use LLM agent,
    # but structured output works the same way with random agent.
    for task_name, TaskCls in TASKS:
        run_task(task_name, TaskCls, seed=seed)


if __name__ == "__main__":
    main()
