

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask
from utils.metrics import (
    EpisodeMetrics,
    compute_episode_metrics,
    compute_utilization,
    compute_waste,
    compute_yield,
    print_metrics_table,
)
from utils.visualization import FabricVisualizer


# ============================================================
# Random Agent Runner
# ============================================================

def run_random_episode(
    env: ZeroWasteFabricEnv,
    seed: int = 42,
    max_steps: int = 500,
) -> Tuple[EpisodeMetrics, List[Dict[str, Any]]]:
    """Run a single episode with a purely random agent.

    Args:
        env: The ZeroWasteFabricEnv to run on.
        seed: Random seed for reproducibility.
        max_steps: Maximum steps to run (caps env's own max).

    Returns:
        Tuple of (EpisodeMetrics, step_history).
    """
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0

    while not terminated and not truncated and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    history = env.get_episode_history()
    metrics = compute_episode_metrics(history)
    return metrics, history


def run_multiple_episodes(
    env: ZeroWasteFabricEnv,
    n_episodes: int = 5,
    seed_start: int = 0,
) -> List[EpisodeMetrics]:
    """Run multiple random episodes and return all metrics.

    Args:
        env: The environment to run on.
        n_episodes: Number of episodes to run.
        seed_start: Starting seed (incremented per episode).

    Returns:
        List of EpisodeMetrics, one per episode.
    """
    all_metrics = []
    for i in range(n_episodes):
        metrics, _ = run_random_episode(env, seed=seed_start + i)
        all_metrics.append(metrics)
    return all_metrics


# ============================================================
# Visualization and Saving
# ============================================================

def save_episode_visualization(
    env: ZeroWasteFabricEnv,
    save_path: str,
    task_name: str,
    metrics: EpisodeMetrics,
) -> None:
    """Save the final state visualization of an episode.

    Args:
        env: The environment after the episode.
        save_path: Path to save the PNG file.
        task_name: Task name for the title.
        metrics: Episode metrics for displaying in the title.
    """
    state = env.get_current_state()
    fabric_length = max(
        env.fabric_max_length,
        state.fabric_length_used + 20,
    )

    title = (
        f"ZeroWaste-Pattern [{task_name}] — Random Agent Rollout\n"
        f"Utilization: {metrics.utilization_pct:.1f}%  |  "
        f"Waste: {metrics.waste_pct:.1f}%  |  "
        f"Placements: {metrics.placements}  |  "
        f"Invalid: {metrics.invalid_actions}  |  "
        f"Steps: {metrics.steps}"
    )

    viz = FabricVisualizer(state, env.fabric_width, fabric_length)
    fig = viz.render(
        save_path=save_path,
        show=False,
        title=title,
    )
    plt.close(fig)
    print(f"  Visualization saved: {save_path}")


def save_metrics_plot(
    history: List[Dict[str, Any]],
    save_path: str,
    task_name: str,
) -> None:
    """Save a metrics plot for an episode.

    Args:
        history: Step history from the episode.
        save_path: Path to save the PNG file.
        task_name: Task name for the title.
    """
    if not history:
        return

    # Create dummy env_state for visualizer (we just need the plot method)
    import numpy as np
    from models.state import EnvironmentState
    dummy_state = EnvironmentState(
        occupancy_grid=np.zeros((10, 10)),
        fabric_width=150.0,
        fabric_max_length=300.0,
    )
    viz = FabricVisualizer(dummy_state, 150.0, 300.0)
    fig = viz.plot_metrics(
        metrics_history=history,
        save_path=save_path,
        show=False,
    )
    plt.close(fig)
    print(f"  Metrics plot saved: {save_path}")


# ============================================================
# Main Demo
# ============================================================

def main() -> None:
    """Run the complete random agent rollout demo."""
    print("=" * 70)
    print("ZeroWaste-Pattern: Random Agent Rollout Demo")
    print("=" * 70)

    output_dir = "rollout_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Define tasks
    tasks_config = [
        {
            "name": "BasicPacking",
            "env_id": "basic",
            "env": ZeroWasteFabricEnv(
                task=BasicPackingTask(), cell_size=2.0, max_steps=300, verbose=False
            ),
            "viz_path": os.path.join(output_dir, "rollout_basic.png"),
            "metrics_path": os.path.join(output_dir, "metrics_basic.png"),
        },
        {
            "name": "IrregularShapes",
            "env_id": "irregular",
            "env": ZeroWasteFabricEnv(
                task=IrregularShapesTask(), cell_size=2.0, max_steps=400, verbose=False
            ),
            "viz_path": os.path.join(output_dir, "rollout_irregular.png"),
            "metrics_path": os.path.join(output_dir, "metrics_irregular.png"),
        },
        {
            "name": "IndustrialMode",
            "env_id": "industrial",
            "env": ZeroWasteFabricEnv(
                task=IndustrialModeTask(), cell_size=2.0, max_steps=500, verbose=False
            ),
            "viz_path": os.path.join(output_dir, "rollout_industrial.png"),
            "metrics_path": os.path.join(output_dir, "metrics_industrial.png"),
        },
    ]

    all_metrics = []
    all_labels = []
    all_histories = []

    # ---- Run one episode per task ----
    for cfg in tasks_config:
        task_name = cfg["name"]
        env: ZeroWasteFabricEnv = cfg["env"]

        print(f"\n{'='*70}")
        print(f"TASK: {task_name}")
        print(f"{'='*70}")
        print(f"  Environment: {env}")
        print(f"  Running random agent episode...")

        start_time = time.time()
        metrics, history = run_random_episode(env, seed=42)
        elapsed = time.time() - start_time

        all_metrics.append(metrics)
        all_labels.append(task_name)
        all_histories.append(history)

        # Print detailed metrics
        print(f"\n  Episode Results ({elapsed:.2f}s):")
        print(f"  {metrics}")

        # Detailed breakdown
        print(f"\n  Detailed Breakdown:")
        print(f"    Steps taken:         {metrics.steps:>8}")
        print(f"    Successful placements:{metrics.placements:>7}")
        print(f"    Pieces placed:       {metrics.pieces_placed:>8} / {metrics.pieces_total}")
        print(f"    Invalid actions:     {metrics.invalid_actions:>8} ({metrics.invalid_actions / max(metrics.steps, 1) * 100:.1f}% of steps)")
        print(f"    Fabric utilization:  {metrics.utilization_pct:>7.2f}%")
        print(f"    Fabric waste:        {metrics.waste_pct:>7.2f}%")
        print(f"    Marker yield:        {metrics.marker_yield_pct:>7.2f}%")
        print(f"    Fabric length used:  {metrics.fabric_length_used:>7.1f} cm")
        print(f"    Total reward:        {metrics.total_reward:>8.4f}")

        # Save visualizations
        print(f"\n  Saving visualizations...")
        try:
            save_episode_visualization(
                env=env,
                save_path=cfg["viz_path"],
                task_name=task_name,
                metrics=metrics,
            )
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")

        try:
            save_metrics_plot(
                history=history,
                save_path=cfg["metrics_path"],
                task_name=task_name,
            )
        except Exception as e:
            print(f"  Warning: Metrics plot failed: {e}")

    # ---- Cross-Task Comparison Table ----
    print("\n" + "=" * 70)
    print("CROSS-TASK COMPARISON: Random Agent (1 episode each)")
    print("=" * 70)
    print_metrics_table(all_metrics, all_labels)

    # ---- Step-by-step Analysis for BasicPacking ----
    print("\n" + "=" * 70)
    print("STEP-BY-STEP ANALYSIS: BasicPacking (first 20 steps)")
    print("=" * 70)

    basic_history = all_histories[0]
    print(f"{'Step':>6} {'Valid':>6} {'Placed':>7} {'Reward':>9} {'Util%':>8} {'Fabric_L':>10}")
    print("-" * 55)
    for step_data in basic_history[:20]:
        valid_str = "YES" if step_data.get("valid", False) else "NO "
        placed_str = "YES" if step_data.get("placed", False) else "NO "
        print(
            f"{step_data.get('step', 0):>6} "
            f"{valid_str:>6} "
            f"{placed_str:>7} "
            f"{step_data.get('reward', 0.0):>9.4f} "
            f"{step_data.get('utilization', 0.0) * 100:>7.2f}% "
            f"{step_data.get('fabric_length_used', 0.0):>9.1f}cm"
        )
    if len(basic_history) > 20:
        print(f"  ... ({len(basic_history) - 20} more steps not shown)")

    # ---- Multi-Episode Statistics for BasicPacking ----
    print("\n" + "=" * 70)
    print("MULTI-EPISODE STATISTICS: BasicPacking (5 episodes)")
    print("=" * 70)

    basic_env = tasks_config[0]["env"]
    multi_metrics = run_multiple_episodes(basic_env, n_episodes=5, seed_start=0)

    utils = [m.utilization_pct for m in multi_metrics]
    wastes = [m.waste_pct for m in multi_metrics]
    rewards = [m.total_reward for m in multi_metrics]
    placements = [m.placements for m in multi_metrics]
    invalids = [m.invalid_actions for m in multi_metrics]

    print_metrics_table(multi_metrics, [f"Episode {i+1}" for i in range(len(multi_metrics))])

    print(f"\n  Aggregated Statistics (N=5):")
    print(f"    Utilization:  mean={np.mean(utils):.2f}%  std={np.std(utils):.2f}%  "
          f"min={np.min(utils):.2f}%  max={np.max(utils):.2f}%")
    print(f"    Waste:        mean={np.mean(wastes):.2f}%  std={np.std(wastes):.2f}%  "
          f"min={np.min(wastes):.2f}%  max={np.max(wastes):.2f}%")
    print(f"    Total Reward: mean={np.mean(rewards):.4f}  std={np.std(rewards):.4f}")
    print(f"    Placements:   mean={np.mean(placements):.1f}  std={np.std(placements):.1f}")
    print(f"    Invalid Acts: mean={np.mean(invalids):.1f}  std={np.std(invalids):.1f}")

    print(f"\n  Output files saved to: {os.path.abspath(output_dir)}/")
    print("\n" + "=" * 70)
    print("Random Agent Rollout Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
