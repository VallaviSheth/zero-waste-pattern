"""Example usage of ZeroWaste-Pattern environment with PPO training.

Demonstrates:
1. Environment creation for all 3 task variants
2. PPO-compatible training loop (using stable-baselines3 if available)
3. Manual training loop fallback
4. Saving and loading trained models
5. Rendering environment state

Run:
    python example_usage.py

Requirements (optional for SB3 training):
    pip install stable-baselines3 gymnasium
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask
from utils.metrics import EpisodeMetrics, compute_episode_metrics, print_metrics_table


# ============================================================
# Part 1: Environment Creation
# ============================================================

def create_all_environments(verbose: bool = False) -> Dict[str, ZeroWasteFabricEnv]:
    """Create all three environment variants.

    Args:
        verbose: Whether to enable verbose logging in environments.

    Returns:
        Dictionary mapping environment names to ZeroWasteFabricEnv instances.
    """
    print("\n" + "=" * 60)
    print("PART 1: Creating ZeroWaste-Pattern Environments")
    print("=" * 60)

    envs = {}

    # Task 1: Basic Packing (rectangles, no grain constraints)
    task1 = BasicPackingTask()
    env1 = ZeroWasteFabricEnv(task=task1, cell_size=2.0, max_steps=300, verbose=verbose)
    envs["basic"] = env1
    print(f"\n[1] {env1}")
    print(f"    Task info: {task1.get_info()}")

    # Task 2: Irregular Shapes (shirt polygons + grain)
    task2 = IrregularShapesTask()
    env2 = ZeroWasteFabricEnv(task=task2, cell_size=2.0, max_steps=400, verbose=verbose)
    envs["irregular"] = env2
    print(f"\n[2] {env2}")
    print(f"    Task info: {task2.get_info()}")

    # Task 3: Industrial Mode (strict constraints + rolling fabric)
    task3 = IndustrialModeTask()
    env3 = ZeroWasteFabricEnv(task=task3, cell_size=2.0, max_steps=500, verbose=verbose)
    envs["industrial"] = env3
    print(f"\n[3] {env3}")
    print(f"    Task info: {task3.get_info()}")

    return envs


# ============================================================
# Part 2: Manual Training Loop (no SB3 dependency)
# ============================================================

def manual_training_loop(
    env: ZeroWasteFabricEnv,
    n_episodes: int = 5,
    n_steps: int = 100,
    env_name: str = "env",
) -> List[EpisodeMetrics]:
    """Run a simple manual training loop (random policy as placeholder).

    This demonstrates the Gymnasium API without requiring SB3.
    Replace the random action selection with your own policy.

    Args:
        env: The ZeroWasteFabricEnv instance to train on.
        n_episodes: Number of episodes to run.
        n_steps: Maximum steps per episode (overrides env.max_steps if provided).
        env_name: Name for display purposes.

    Returns:
        List of EpisodeMetrics from all episodes.
    """
    print(f"\n--- Manual Training Loop: {env_name} ({n_episodes} episodes) ---")

    all_metrics = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and step < n_steps:
            # Random action (replace with learned policy)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        # Collect metrics
        history = env.get_episode_history()
        metrics = compute_episode_metrics(history)
        all_metrics.append(metrics)

        print(
            f"  Episode {episode+1:2d}: "
            f"util={metrics.utilization_pct:5.1f}% "
            f"waste={metrics.waste_pct:5.1f}% "
            f"reward={episode_reward:7.3f} "
            f"placements={metrics.placements:3d} "
            f"invalid={metrics.invalid_actions:3d} "
            f"steps={step:4d}"
        )

    return all_metrics


# ============================================================
# Part 3: SB3 Training (if available)
# ============================================================

def try_sb3_training(
    env: ZeroWasteFabricEnv,
    n_timesteps: int = 10000,
    save_path: Optional[str] = None,
) -> bool:
    """Attempt to run a short PPO training session using stable-baselines3.

    Args:
        env: The environment to train on.
        n_timesteps: Number of training timesteps.
        save_path: Optional path to save the trained model.

    Returns:
        True if training succeeded, False if SB3 is not available.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor

        print("\n" + "=" * 60)
        print("PART 3: SB3 PPO Training")
        print("=" * 60)

        # Wrap with Monitor for episode stats
        monitored_env = Monitor(env)

        print("\nRunning environment sanity check...")
        try:
            check_env(env, warn=True, skip_render_check=True)
            print("  Environment check PASSED")
        except Exception as e:
            print(f"  Environment check WARNING: {e}")

        print(f"\nCreating PPO model (n_timesteps={n_timesteps})...")
        model = PPO(
            policy="MultiInputPolicy",
            env=monitored_env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )

        print(f"Training for {n_timesteps} timesteps...")
        start_time = time.time()
        model.learn(total_timesteps=n_timesteps)
        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.1f}s")

        if save_path:
            model.save(save_path)
            print(f"Model saved to: {save_path}")

        # Run one evaluation episode
        print("\nRunning evaluation episode with trained policy...")
        obs, info = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        print(f"  Evaluation: reward={episode_reward:.3f}, "
              f"util={info['utilization_pct']:.1f}%, "
              f"steps={steps}")

        return True

    except ImportError:
        print("\n[INFO] stable-baselines3 not installed. Skipping SB3 training.")
        print("       Install with: pip install stable-baselines3")
        return False


# ============================================================
# Part 4: Model Loading Example
# ============================================================

def demonstrate_model_loading(model_path: str, env: ZeroWasteFabricEnv) -> None:
    """Demonstrate loading a saved model and running inference.

    Args:
        model_path: Path to the saved SB3 model file.
        env: The environment to run inference on.
    """
    try:
        from stable_baselines3 import PPO

        print(f"\nLoading model from: {model_path}")
        model = PPO.load(model_path, env=env)

        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(f"Loaded model episode: reward={total_reward:.3f}, "
              f"util={info['utilization_pct']:.1f}%")

    except ImportError:
        print("[INFO] stable-baselines3 not available for model loading demo.")
    except FileNotFoundError:
        print(f"[INFO] No saved model at {model_path}. Train first.")


# ============================================================
# Part 5: Rendering Example
# ============================================================

def demonstrate_rendering(env: ZeroWasteFabricEnv, n_steps: int = 20) -> None:
    """Demonstrate rendering the environment after some random placements.

    Args:
        env: The environment to render.
        n_steps: Number of random steps before rendering.
    """
    print("\n--- Demonstrating Rendering ---")

    obs, info = env.reset(seed=42)

    # Take some random steps
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(f"  After {n_steps} random steps:")
    print(f"  Utilization: {info['utilization_pct']:.1f}%")
    print(f"  Pieces placed: {info['pieces_placed']}")

    # Render (saves to file to avoid display issues in headless environments)
    try:
        state = env.get_current_state()
        from utils.visualization import FabricVisualizer
        viz = FabricVisualizer(
            state,
            env.fabric_width,
            max(env.fabric_max_length, state.fabric_length_used + 10),
        )
        save_path = "render_demo.png"
        fig = viz.render(save_path=save_path, show=False)
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Rendered state saved to: {save_path}")
    except Exception as e:
        print(f"  Rendering error (non-critical): {e}")


# ============================================================
# Main Entry Point
# ============================================================

def main() -> None:
    """Run all example usage demonstrations."""
    print("=" * 60)
    print("ZeroWaste-Pattern: The AI Sustainable Tailor")
    print("Example Usage Script")
    print("=" * 60)

    # Part 1: Create environments
    envs = create_all_environments(verbose=False)

    # Part 2: Manual training loops for each task
    all_metrics: Dict[str, List[EpisodeMetrics]] = {}

    for env_name, env in envs.items():
        metrics_list = manual_training_loop(
            env=env,
            n_episodes=3,
            n_steps=200,
            env_name=env_name,
        )
        all_metrics[env_name] = metrics_list

    # Print comparison table
    print("\n" + "=" * 60)
    print("SUMMARY: Final Episode Metrics per Task (last of each)")
    print("=" * 60)

    final_metrics = [m_list[-1] for m_list in all_metrics.values()]
    labels = [f"Task: {name}" for name in all_metrics.keys()]
    print_metrics_table(final_metrics, labels)

    # Part 3: SB3 Training (basic task, short run)
    basic_env = envs["basic"]
    model_path = "ppo_zerowaste_basic"
    trained = try_sb3_training(
        env=basic_env,
        n_timesteps=2000,
        save_path=model_path,
    )

    # Part 4: Load and run if model was saved
    if trained and os.path.exists(f"{model_path}.zip"):
        demonstrate_model_loading(model_path, basic_env)

    # Part 5: Rendering demo
    demonstrate_rendering(envs["basic"], n_steps=30)

    print("\n" + "=" * 60)
    print("Example usage complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
