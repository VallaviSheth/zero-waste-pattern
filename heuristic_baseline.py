"""Heuristic baseline for ZeroWaste-Pattern environment.

Implements a Greedy Bottom-Left packer as a deterministic baseline.
Compares its performance against a random agent across all three tasks.

The greedy algorithm:
1. Sorts pieces by area (largest first — better space utilization).
2. For each piece, scans positions from bottom-left to top-right.
3. For each candidate position, tries all allowed rotations.
4. Places the piece at the first valid position found.
5. Continues until all pieces are placed or no valid position exists.

Run:
    python heuristic_baseline.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from env.fabric_space import FabricSpace
from models.action import PlacementAction
from models.pattern_piece import PatternPiece
from models.state import EnvironmentState, PlacedPiece
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask
from utils.geometry import (
    check_no_overlap,
    check_within_bounds,
    compute_fragmentation,
    place_polygon,
)
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
# Greedy Bottom-Left Packer
# ============================================================

class GreedyBottomLeftPacker:
    """Greedy Bottom-Left packing heuristic for garment marker making.

    Sorts pieces by area (largest first) and places each piece at the
    bottom-left-most valid position, trying all allowed rotations.

    This is a classic bin-packing heuristic that provides a reasonable
    baseline for the RL agent to beat.

    Attributes:
        fabric_width: Width of the fabric in cm.
        fabric_max_length: Maximum length of the fabric in cm.
        cell_size: Scanning step size in cm.
        rolling_fabric: Whether to support rolling fabric extension.
    """

    def __init__(
        self,
        fabric_width: float,
        fabric_max_length: float,
        cell_size: float = 2.0,
        rolling_fabric: bool = False,
    ) -> None:
        """Initialize the greedy packer.

        Args:
            fabric_width: Fabric width in cm.
            fabric_max_length: Maximum fabric length in cm.
            cell_size: Step size for position scanning in cm.
            rolling_fabric: Whether rolling fabric extension is enabled.
        """
        self.fabric_width = fabric_width
        self.fabric_max_length = fabric_max_length
        self.cell_size = cell_size
        self.rolling_fabric = rolling_fabric

        self._fabric_space = FabricSpace(
            width=fabric_width,
            max_length=fabric_max_length,
            cell_size=cell_size,
            rolling_fabric=rolling_fabric,
        )
        self.placed_pieces: List[PlacedPiece] = []

    def reset(self) -> None:
        """Reset the packer for a new run."""
        self._fabric_space.reset()
        self.placed_pieces = []

    def pack(self, pieces: List[PatternPiece]) -> Tuple[List[PlacedPiece], Dict[str, Any]]:
        """Run the greedy bottom-left packing algorithm.

        Args:
            pieces: List of PatternPiece instances to pack.
                    Note: pieces with quantity > 1 are expanded automatically.

        Returns:
            Tuple of (placed_pieces, metrics_dict).
        """
        self.reset()

        # Expand pieces by quantity
        expanded = self._expand_pieces(pieces)

        # Sort by area, largest first (better utilization)
        sorted_pieces = sorted(expanded, key=lambda p: p.area, reverse=True)

        n_placed = 0
        n_failed = 0

        for piece in sorted_pieces:
            placed = self._try_place_piece(piece)
            if placed:
                n_placed += 1
            else:
                n_failed += 1

        # Compute final metrics
        placed_polys = [pp.get_polygon() for pp in self.placed_pieces]
        length_used = self._fabric_space.get_bounding_used_length()
        length_used = max(length_used, 1.0)

        utilization = compute_utilization(placed_polys, self.fabric_width, length_used)
        total_area = sum(p.area for p in expanded)
        yield_frac = compute_yield(placed_polys, total_area) if total_area > 0 else 0.0

        metrics = {
            "pieces_placed": n_placed,
            "pieces_failed": n_failed,
            "pieces_total": len(expanded),
            "utilization_pct": utilization * 100.0,
            "waste_pct": (1.0 - utilization) * 100.0,
            "marker_yield_pct": yield_frac * 100.0,
            "fabric_length_used": length_used,
        }

        return list(self.placed_pieces), metrics

    def _try_place_piece(self, piece: PatternPiece) -> bool:
        """Try to place a single piece at the best available bottom-left position.

        Scans positions from (0, 0) to (fabric_width, current_length),
        trying all allowed rotations at each position.

        Args:
            piece: The PatternPiece to place.

        Returns:
            True if the piece was successfully placed, False otherwise.
        """
        best_x: Optional[float] = None
        best_y: Optional[float] = None
        best_rotation: Optional[float] = None

        # Determine effective scan length
        current_length = self._fabric_space.current_length
        if self.rolling_fabric:
            scan_length = current_length + 100.0  # allow some extension
        else:
            scan_length = current_length

        # Scan positions: bottom to top, left to right.
        # Because we scan BL→TR and want the lowest-y, lowest-x valid placement,
        # the FIRST valid placement found is already the optimal BL position.
        found = False
        y = 0.0
        while y < scan_length and not found:
            x = 0.0
            while x < self.fabric_width and not found:
                # Try all allowed rotations at this position
                for rotation in piece.allowed_rotations:
                    # Check grain constraint
                    if not piece.check_grain_alignment(rotation):
                        continue

                    try:
                        poly = place_polygon(piece.polygon, x, y, rotation)
                    except Exception:
                        continue

                    _minx, _miny, maxx, maxy = poly.bounds

                    # Check width bounds
                    if maxx > self.fabric_width + 1e-4:
                        continue

                    # Check length bounds (for non-rolling)
                    if not self.rolling_fabric and maxy > self.fabric_max_length + 1e-4:
                        continue

                    # Check overlap
                    is_valid, _ = self._fabric_space.is_valid_placement(poly)
                    if not is_valid:
                        continue

                    # First valid BL position — take it
                    best_x = x
                    best_y = y
                    best_rotation = rotation
                    found = True
                    break

                x += self.cell_size
            y += self.cell_size

        if best_x is None:
            return False

        # Place at best position
        try:
            placed_poly = place_polygon(piece.polygon, best_x, best_y, best_rotation)

            # Extend fabric if needed (rolling mode)
            _, _, _, maxy = placed_poly.bounds
            if self.rolling_fabric and maxy > self._fabric_space.current_length:
                extension = max(50.0, np.ceil((maxy - self._fabric_space.current_length) / 50.0) * 50.0)
                self._fabric_space.extend_fabric(extension)

            success = self._fabric_space.place_piece(placed_poly)
            if not success:
                return False

            placed = PlacedPiece(
                piece=piece,
                x=best_x,
                y=best_y,
                rotation_deg=best_rotation,
                placed_polygon=placed_poly,
                placement_step=len(self.placed_pieces),
            )
            self.placed_pieces.append(placed)
            return True

        except Exception:
            return False

    @staticmethod
    def _expand_pieces(pieces: List[PatternPiece]) -> List[PatternPiece]:
        """Expand pieces by quantity into individual instances.

        Args:
            pieces: List of PatternPiece instances.

        Returns:
            Expanded list with quantity=1 per instance.
        """
        expanded = []
        for piece in pieces:
            if piece.quantity == 1:
                expanded.append(piece)
            else:
                for q in range(piece.quantity):
                    new_piece = PatternPiece(
                        id=f"{piece.id}_copy{q+1}",
                        name=piece.name,
                        vertices=piece.vertices,
                        grain_direction=piece.grain_direction,
                        grain_tolerance_deg=piece.grain_tolerance_deg,
                        allowed_rotations=piece.allowed_rotations,
                        quantity=1,
                        color=piece.color,
                    )
                    expanded.append(new_piece)
        return expanded

    def get_current_state(self, fabric_max_length: float) -> EnvironmentState:
        """Build an EnvironmentState from the packer's current state.

        Args:
            fabric_max_length: Maximum fabric length for state construction.

        Returns:
            EnvironmentState representing the packed layout.
        """
        length_used = self._fabric_space.get_bounding_used_length()
        occupancy = self._fabric_space.occupancy_grid.copy()

        placed_polys = [pp.get_polygon() for pp in self.placed_pieces]
        length_used_actual = max(length_used, 1.0)
        utilization = compute_utilization(placed_polys, self.fabric_width, length_used_actual)

        return EnvironmentState(
            occupancy_grid=occupancy,
            remaining_pieces=[],
            placed_pieces=self.placed_pieces,
            utilization_pct=utilization * 100.0,
            fabric_width=self.fabric_width,
            fabric_length_used=length_used,
            fabric_max_length=fabric_max_length,
            step_count=len(self.placed_pieces),
            episode_done=True,
            invalid_action_count=0,
            total_reward=0.0,
        )


# ============================================================
# Random Agent Runner (for comparison)
# ============================================================

def run_best_random_episode(
    env: ZeroWasteFabricEnv,
    n_episodes: int = 5,
    seed_start: int = 0,
) -> Tuple[EpisodeMetrics, int]:
    """Run N random episodes and return the best result.

    Args:
        env: The environment to run on.
        n_episodes: Number of episodes to attempt.
        seed_start: Starting random seed.

    Returns:
        Tuple of (best_EpisodeMetrics, best_episode_index).
    """
    best_metrics: Optional[EpisodeMetrics] = None
    best_idx = 0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + i)
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        history = env.get_episode_history()
        metrics = compute_episode_metrics(history)

        if best_metrics is None or metrics.utilization_pct > best_metrics.utilization_pct:
            best_metrics = metrics
            best_idx = i

    return best_metrics, best_idx


# ============================================================
# Main Comparison
# ============================================================

def run_comparison() -> None:
    """Run the full greedy vs random comparison across all three tasks.

    For each task:
    1. Runs the greedy bottom-left packer once (deterministic)
    2. Runs the random agent N episodes and takes the best
    3. Prints a comparison table
    4. Saves visualizations
    """
    output_dir = "heuristic_outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ZeroWaste-Pattern: Heuristic vs Random Agent Comparison")
    print("=" * 80)

    RANDOM_EPISODES = 5  # Number of random episodes to run (take best)

    tasks_config = [
        {
            "name": "BasicPacking",
            "task_cls": BasicPackingTask,
            "cell_size": 2.0,
        },
        {
            "name": "IrregularShapes",
            "task_cls": IrregularShapesTask,
            "cell_size": 2.0,
        },
        {
            "name": "IndustrialMode",
            "task_cls": IndustrialModeTask,
            "cell_size": 2.0,
        },
    ]

    comparison_rows = []
    comparison_labels = []

    for cfg in tasks_config:
        task_name = cfg["name"]
        print(f"\n{'='*80}")
        print(f"TASK: {task_name}")
        print(f"{'='*80}")

        # ---- Instantiate task ----
        task = cfg["task_cls"]()
        fabric_w, fabric_l = task.get_fabric_dimensions()
        pieces = task.get_pieces()
        rolling = task.supports_rolling_fabric()

        print(f"  Fabric: {fabric_w} x {fabric_l} cm  |  "
              f"Pieces: {len(pieces)} types / {sum(p.quantity for p in pieces)} instances  |  "
              f"Rolling: {rolling}")

        # ---- Run Greedy ----
        print(f"\n  Running Greedy Bottom-Left Packer...")
        greedy_start = time.time()

        packer = GreedyBottomLeftPacker(
            fabric_width=fabric_w,
            fabric_max_length=fabric_l,
            cell_size=cfg["cell_size"],
            rolling_fabric=rolling,
        )
        placed_pieces, greedy_stats = packer.pack(pieces)
        greedy_elapsed = time.time() - greedy_start

        greedy_metrics = EpisodeMetrics(
            steps=greedy_stats["pieces_total"],
            placements=greedy_stats["pieces_placed"],
            utilization_pct=greedy_stats["utilization_pct"],
            waste_pct=greedy_stats["waste_pct"],
            marker_yield_pct=greedy_stats["marker_yield_pct"],
            total_reward=0.0,  # Greedy doesn't compute RL reward
            invalid_actions=0,
            fabric_length_used=greedy_stats["fabric_length_used"],
            pieces_placed=greedy_stats["pieces_placed"],
            pieces_total=greedy_stats["pieces_total"],
        )

        print(f"  Greedy Results ({greedy_elapsed:.2f}s):")
        print(f"    Pieces placed:      {greedy_stats['pieces_placed']} / {greedy_stats['pieces_total']}")
        print(f"    Utilization:        {greedy_stats['utilization_pct']:.2f}%")
        print(f"    Waste:              {greedy_stats['waste_pct']:.2f}%")
        print(f"    Marker yield:       {greedy_stats['marker_yield_pct']:.2f}%")
        print(f"    Fabric length used: {greedy_stats['fabric_length_used']:.1f} cm")

        # Save greedy visualization
        try:
            greedy_state = packer.get_current_state(fabric_l)
            viz_greedy = FabricVisualizer(
                greedy_state,
                fabric_w,
                max(fabric_l, greedy_stats["fabric_length_used"] + 20),
            )
            greedy_viz_path = os.path.join(output_dir, f"greedy_{task_name.lower()}.png")
            fig = viz_greedy.render(
                save_path=greedy_viz_path,
                show=False,
                title=(
                    f"Greedy BL Packer [{task_name}]\n"
                    f"Utilization: {greedy_stats['utilization_pct']:.1f}%  |  "
                    f"Pieces: {greedy_stats['pieces_placed']}/{greedy_stats['pieces_total']}"
                ),
            )
            plt.close(fig)
            print(f"    Visualization saved: {greedy_viz_path}")
        except Exception as e:
            print(f"    Warning: Greedy visualization failed: {e}")

        # ---- Run Random Agent ----
        print(f"\n  Running Random Agent ({RANDOM_EPISODES} episodes, taking best)...")
        random_start = time.time()

        env = ZeroWasteFabricEnv(task=task, cell_size=cfg["cell_size"], verbose=False)
        best_random_metrics, best_ep = run_best_random_episode(
            env, n_episodes=RANDOM_EPISODES, seed_start=0
        )
        random_elapsed = time.time() - random_start

        print(f"  Random Agent Best Results ({random_elapsed:.2f}s, episode {best_ep+1}):")
        print(f"    Pieces placed:      {best_random_metrics.pieces_placed} / {best_random_metrics.pieces_total}")
        print(f"    Utilization:        {best_random_metrics.utilization_pct:.2f}%")
        print(f"    Waste:              {best_random_metrics.waste_pct:.2f}%")
        print(f"    Marker yield:       {best_random_metrics.marker_yield_pct:.2f}%")
        print(f"    Fabric length used: {best_random_metrics.fabric_length_used:.1f} cm")
        print(f"    Total reward:       {best_random_metrics.total_reward:.4f}")
        print(f"    Invalid actions:    {best_random_metrics.invalid_actions}")

        # Save random agent visualization
        try:
            # Re-run the best episode to get the state
            obs, _ = env.reset(seed=best_ep)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

            random_state = env.get_current_state()
            viz_random = FabricVisualizer(
                random_state,
                fabric_w,
                max(fabric_l, random_state.fabric_length_used + 20),
            )
            random_viz_path = os.path.join(output_dir, f"random_{task_name.lower()}.png")
            fig = viz_random.render(
                save_path=random_viz_path,
                show=False,
                title=(
                    f"Random Agent [{task_name}]\n"
                    f"Utilization: {best_random_metrics.utilization_pct:.1f}%  |  "
                    f"Pieces: {best_random_metrics.pieces_placed}/{best_random_metrics.pieces_total}"
                ),
            )
            plt.close(fig)
            print(f"    Visualization saved: {random_viz_path}")
        except Exception as e:
            print(f"    Warning: Random visualization failed: {e}")

        # ---- Comparison ----
        util_improvement = greedy_stats["utilization_pct"] - best_random_metrics.utilization_pct
        print(f"\n  COMPARISON: Greedy vs Random")
        print(f"    Utilization improvement: {util_improvement:+.2f}%")
        print(f"    Additional pieces placed: {greedy_stats['pieces_placed'] - best_random_metrics.pieces_placed:+d}")

        comparison_rows.append(greedy_metrics)
        comparison_rows.append(best_random_metrics)
        comparison_labels.append(f"[Greedy] {task_name}")
        comparison_labels.append(f"[Random] {task_name}")

    # ---- Full Comparison Table ----
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE: Greedy Bottom-Left vs Random Agent")
    print("=" * 80)
    print_metrics_table(comparison_rows, comparison_labels)

    # ---- Per-Task Summary ----
    print("\n" + "=" * 80)
    print("PER-TASK SUMMARY")
    print("=" * 80)
    print(f"\n{'Task':<20} {'Greedy Util':>15} {'Random Util':>15} {'Improvement':>15}")
    print("-" * 70)

    task_names = [cfg["name"] for cfg in tasks_config]
    for i, name in enumerate(task_names):
        g_metrics = comparison_rows[i * 2]
        r_metrics = comparison_rows[i * 2 + 1]
        improvement = g_metrics.utilization_pct - r_metrics.utilization_pct
        print(f"{name:<20} {g_metrics.utilization_pct:>14.2f}% {r_metrics.utilization_pct:>14.2f}% {improvement:>+14.2f}%")

    print("\n" + "=" * 80)
    print(f"Output files saved to: {os.path.abspath(output_dir)}/")
    print("\nGreedy Bottom-Left Packer provides a strong deterministic baseline.")
    print("A well-trained RL agent should exceed greedy performance by learning")
    print("to consider future placements and rotational strategies.")
    print("=" * 80)


# ============================================================
# Additional Greedy Analysis
# ============================================================

def analyze_greedy_sensitivity() -> None:
    """Analyze sensitivity of greedy packer to piece ordering.

    Tests multiple orderings (by area, by width, by height, random)
    on the BasicPacking task and compares results.
    """
    print("\n" + "=" * 80)
    print("GREEDY SENSITIVITY ANALYSIS: BasicPacking Task")
    print("=" * 80)

    task = BasicPackingTask()
    pieces = task.get_pieces()
    fabric_w, fabric_l = task.get_fabric_dimensions()

    # Expand pieces
    expanded = GreedyBottomLeftPacker._expand_pieces(pieces)

    orderings = {
        "by_area_desc": sorted(expanded, key=lambda p: p.area, reverse=True),
        "by_area_asc": sorted(expanded, key=lambda p: p.area),
        "by_width_desc": sorted(expanded, key=lambda p: p.get_natural_width(), reverse=True),
        "by_height_desc": sorted(expanded, key=lambda p: p.get_natural_height(), reverse=True),
        "original_order": expanded,
    }

    for name, ordered_pieces in orderings.items():
        packer = GreedyBottomLeftPacker(
            fabric_width=fabric_w,
            fabric_max_length=fabric_l,
            cell_size=2.0,
        )
        # Manually run with the given order (bypass sort in pack())
        packer.reset()
        n_placed = 0
        for piece in ordered_pieces:
            placed = packer._try_place_piece(piece)
            if placed:
                n_placed += 1

        placed_polys = [pp.get_polygon() for pp in packer.placed_pieces]
        length_used = max(packer._fabric_space.get_bounding_used_length(), 1.0)
        util = compute_utilization(placed_polys, fabric_w, length_used)

        print(f"  Ordering '{name:<20}': "
              f"util={util*100:>6.2f}%  "
              f"placed={n_placed}/{len(expanded)}  "
              f"length_used={length_used:>6.1f} cm")


if __name__ == "__main__":
    run_comparison()
    print()
    analyze_greedy_sensitivity()
