"""Irregular shapes task for ZeroWaste-Pattern environment.

Task 2: Real shirt pattern polygons with grain constraints.
Requires the agent to handle non-rectangular shapes and orientation constraints.
"""

from __future__ import annotations

from typing import List, Tuple

from models.pattern_piece import PatternPiece
from models.reward_config import RewardConfig
from models.state import EnvironmentState
from tasks.base_task import BaseTask
from utils.dataset import PatternDataset


class IrregularShapesTask(BaseTask):
    """Shirt pattern cutting task with irregular polygons and grain constraints.

    Uses the full shirt pattern set from the dataset generator on a larger
    150x400 cm fabric. Grain constraints (VERTICAL or HORIZONTAL) apply to
    most pieces, and rotations are limited to 0 and 180 degrees for bodice pieces.

    Fabric: 150 cm wide x 400 cm long
    Pieces: Full shirt set (10 piece types with quantities) = ~20+ instances
    Constraints:
        - Bodice pieces: VERTICAL grain, rotations [0, 180]
        - Collar/cuff/yoke: HORIZONTAL grain, rotations [0, 180]
        - Small accessories: ANY grain, rotations [0, 90, 180, 270]
    Reward: Standard configuration
    """

    def __init__(self) -> None:
        """Initialize the irregular shapes task with the shirt pattern dataset."""
        self._pieces: List[PatternPiece] = PatternDataset.generate_shirt_set()

    @property
    def name(self) -> str:
        """Return the task name."""
        return "IrregularShapes"

    @property
    def description(self) -> str:
        """Return the task description."""
        return (
            "Full shirt pattern cutting task with irregular polygon shapes. "
            "Place shirt pieces (front/back bodice, sleeves, collar, cuffs, yoke, panels) "
            "on 150x400 cm fabric respecting grain direction constraints. "
            "Bodice and sleeve pieces must maintain VERTICAL grain (rotations: 0, 180 deg). "
            "Collar and cuff pieces must maintain HORIZONTAL grain. "
            "Goal: maximize utilization while respecting all constraints."
        )

    def grade(self, state: EnvironmentState) -> float:
        """Grade agent performance on irregular shapes (medium task).

        Scoring breakdown:
        - 50% weight: utilization percentage (target 70%+)
        - 30% weight: piece placement ratio
        - 20% weight: efficiency penalty (fewer invalid actions = better)

        Args:
            state: Final EnvironmentState at episode end.

        Returns:
            Score in [0.0, 1.0].
        """
        util = state.utilization_pct / 100.0
        util_score = min(util / 0.70, 1.0)

        total_pieces = state.n_placed + state.n_remaining
        placement_ratio = state.n_placed / max(total_pieces, 1)

        # Efficiency: penalize high invalid action rates
        total_actions = state.step_count if state.step_count > 0 else 1
        invalid_rate = state.invalid_action_count / total_actions
        efficiency = max(1.0 - invalid_rate, 0.0)

        score = 0.5 * util_score + 0.3 * placement_ratio + 0.2 * efficiency
        return round(min(max(score, 0.0), 1.0), 4)

    def get_pieces(self) -> List[PatternPiece]:
        """Return the full shirt pattern piece set.

        Includes all piece types with their quantities as defined in the
        shirt pattern dataset.

        Returns:
            List of PatternPiece instances for a complete shirt pattern.
        """
        return list(self._pieces)

    def get_fabric_dimensions(self) -> Tuple[float, float]:
        """Return the fabric dimensions for the irregular shapes task.

        Returns:
            Tuple of (150.0, 400.0) for 150 cm wide x 400 cm long.
        """
        return (150.0, 400.0)

    def get_reward_config(self) -> RewardConfig:
        """Return the standard reward configuration for this task.

        Returns:
            Standard RewardConfig with balanced reward shaping.
        """
        return RewardConfig(
            placement_reward_scale=10.0,
            invalid_action_penalty=-0.5,
            overlap_penalty=-1.0,
            out_of_bounds_penalty=-0.8,
            grain_violation_penalty=-0.6,
            completion_bonus_scale=5.0,
            fragmentation_penalty_scale=0.05,
            step_penalty=-0.01,
        )

    def get_max_steps(self) -> int:
        """Return 400 max steps for the irregular shapes task.

        Returns:
            Maximum steps per episode.
        """
        return 400
