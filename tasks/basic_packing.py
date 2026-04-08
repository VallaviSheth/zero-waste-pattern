"""Basic packing task for ZeroWaste-Pattern environment.

Task 1: Simple rectangular pieces with no grain constraints.
Suitable for initial RL training and algorithm development.
"""

from __future__ import annotations

from typing import List, Tuple

from models.pattern_piece import GrainDirection, PatternPiece
from models.reward_config import RewardConfig
from models.state import EnvironmentState
from tasks.base_task import BaseTask


class BasicPackingTask(BaseTask):
    """Simple rectangular packing task with no grain constraints.

    Uses 10-15 rectangular pieces of varying sizes on a 150x300 cm fabric.
    All rotations are allowed and no grain constraints apply, making this
    the simplest task for getting RL algorithms to learn efficient packing.

    Fabric: 150 cm wide x 300 cm long
    Pieces: 12 rectangular pieces with quantities
    Constraints: Any rotation (0, 90, 180, 270 degrees), no grain
    Reward: Lenient configuration to encourage exploration
    """

    @property
    def name(self) -> str:
        """Return the task name."""
        return "BasicPacking"

    @property
    def description(self) -> str:
        """Return the task description."""
        return (
            "Basic rectangular packing task. "
            "Place 12 rectangular pieces on 150x300 cm fabric with no grain constraints. "
            "All 4 rotations (0, 90, 180, 270 degrees) are allowed for all pieces. "
            "Goal: maximize fabric utilization."
        )

    def grade(self, state: EnvironmentState) -> float:
        """Grade agent performance on basic packing (easy task).

        Scoring breakdown (all contribute to 0.0-1.0):
        - 60% weight: utilization percentage (target 75%+)
        - 40% weight: piece placement ratio (placed/total)

        Args:
            state: Final EnvironmentState at episode end.

        Returns:
            Score in [0.0, 1.0].
        """
        # Utilization component (0-1, target 75%)
        util = state.utilization_pct / 100.0  # normalize to 0-1
        util_score = min(util / 0.75, 1.0)

        # Placement ratio component
        total_pieces = state.n_placed + state.n_remaining
        placement_ratio = state.n_placed / max(total_pieces, 1)

        score = 0.6 * util_score + 0.4 * placement_ratio
        return round(min(max(score, 0.0), 1.0), 4)

    def get_pieces(self) -> List[PatternPiece]:
        """Return the list of rectangular pattern pieces for this task.

        Returns:
            List of 12 rectangular PatternPiece instances of varying sizes.
        """
        pieces = [
            PatternPiece(
                id="basic_rect_01",
                name="Large Panel A",
                vertices=[(0.0, 0.0), (45.0, 0.0), (45.0, 65.0), (0.0, 65.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#E74C3C",
            ),
            PatternPiece(
                id="basic_rect_02",
                name="Large Panel B",
                vertices=[(0.0, 0.0), (40.0, 0.0), (40.0, 58.0), (0.0, 58.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#3498DB",
            ),
            PatternPiece(
                id="basic_rect_03",
                name="Medium Panel A",
                vertices=[(0.0, 0.0), (35.0, 0.0), (35.0, 48.0), (0.0, 48.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#2ECC71",
            ),
            PatternPiece(
                id="basic_rect_04",
                name="Medium Panel B",
                vertices=[(0.0, 0.0), (30.0, 0.0), (30.0, 42.0), (0.0, 42.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#F39C12",
            ),
            PatternPiece(
                id="basic_rect_05",
                name="Small Panel A",
                vertices=[(0.0, 0.0), (22.0, 0.0), (22.0, 32.0), (0.0, 32.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#9B59B6",
            ),
            PatternPiece(
                id="basic_rect_06",
                name="Small Panel B",
                vertices=[(0.0, 0.0), (18.0, 0.0), (18.0, 28.0), (0.0, 28.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#1ABC9C",
            ),
            PatternPiece(
                id="basic_rect_07",
                name="Wide Strip",
                vertices=[(0.0, 0.0), (55.0, 0.0), (55.0, 18.0), (0.0, 18.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#E67E22",
            ),
            PatternPiece(
                id="basic_rect_08",
                name="Narrow Strip",
                vertices=[(0.0, 0.0), (12.0, 0.0), (12.0, 48.0), (0.0, 48.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#2980B9",
            ),
            PatternPiece(
                id="basic_rect_09",
                name="Square Block",
                vertices=[(0.0, 0.0), (28.0, 0.0), (28.0, 28.0), (0.0, 28.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#27AE60",
            ),
            PatternPiece(
                id="basic_rect_10",
                name="Tall Panel",
                vertices=[(0.0, 0.0), (20.0, 0.0), (20.0, 60.0), (0.0, 60.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#8E44AD",
            ),
            PatternPiece(
                id="basic_rect_11",
                name="Banner Piece",
                vertices=[(0.0, 0.0), (60.0, 0.0), (60.0, 14.0), (0.0, 14.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#16A085",
            ),
            PatternPiece(
                id="basic_rect_12",
                name="Mini Block",
                vertices=[(0.0, 0.0), (15.0, 0.0), (15.0, 20.0), (0.0, 20.0)],
                grain_direction=GrainDirection.ANY,
                grain_tolerance_deg=45.0,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=3,
                color="#D35400",
            ),
        ]
        return pieces

    def get_fabric_dimensions(self) -> Tuple[float, float]:
        """Return the fabric dimensions for the basic packing task.

        Returns:
            Tuple of (150.0, 300.0) for 150 cm wide x 300 cm long.
        """
        return (150.0, 300.0)

    def get_reward_config(self) -> RewardConfig:
        """Return a lenient reward configuration for the basic task.

        Returns:
            Lenient RewardConfig with reduced penalties for easy exploration.
        """
        return RewardConfig.lenient()

    def get_max_steps(self) -> int:
        """Return 300 max steps for the basic task.

        Returns:
            Maximum steps per episode.
        """
        return 300
