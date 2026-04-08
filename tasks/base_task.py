"""Abstract base task definition for ZeroWaste-Pattern environment.

All task implementations must inherit from BaseTask and implement the
required abstract methods to define pieces, fabric dimensions, and rewards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from models.action import PlacementAction
from models.pattern_piece import PatternPiece
from models.reward_config import RewardConfig
from models.state import EnvironmentState


class BaseTask(ABC):
    """Abstract base class for all ZeroWaste-Pattern tasks.

    Each task defines a specific marker-making challenge with its own
    fabric dimensions, pattern pieces, constraints, and reward shaping.

    Subclasses must implement:
    - get_pieces(): Return the list of pattern pieces for this task.
    - get_fabric_dimensions(): Return (width, max_length) of the fabric.
    - get_reward_config(): Return the reward shaping configuration.
    - name: Property returning the task name string.
    - description: Property returning the task description string.
    """

    @abstractmethod
    def get_pieces(self) -> List[PatternPiece]:
        """Return the list of pattern pieces for this task.

        Pieces with quantity > 1 should be returned as a single piece,
        and the environment will handle quantity tracking.

        Returns:
            List of PatternPiece instances defining the cutting requirements.
        """
        ...

    @abstractmethod
    def get_fabric_dimensions(self) -> Tuple[float, float]:
        """Return the fabric dimensions for this task.

        Returns:
            Tuple of (width_cm, max_length_cm).
        """
        ...

    @abstractmethod
    def get_reward_config(self) -> RewardConfig:
        """Return the reward configuration for this task.

        Returns:
            RewardConfig instance with all reward shaping parameters.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this task.

        Returns:
            Task name string.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this task's objectives and constraints.

        Returns:
            Task description string.
        """
        ...

    @abstractmethod
    def grade(self, state: EnvironmentState) -> float:
        """Grade the agent's performance on this task.

        Returns a score between 0.0 (worst) and 1.0 (perfect).
        Grading is deterministic given the same state.

        Args:
            state: Final EnvironmentState at episode end.

        Returns:
            Score in [0.0, 1.0].
        """
        ...

    def validate_action(
        self,
        action: PlacementAction,
        state: EnvironmentState,
    ) -> Tuple[bool, str]:
        """Validate a placement action against this task's constraints.

        Base implementation checks piece index validity and grain alignment.
        Subclasses can override for additional task-specific validation.

        Args:
            action: The placement action to validate.
            state: The current environment state.

        Returns:
            Tuple of (is_valid, reason_string).
        """
        # Check piece index validity
        if action.piece_index < 0 or action.piece_index >= len(state.remaining_pieces):
            return False, f"Invalid piece index {action.piece_index} (have {len(state.remaining_pieces)} remaining)"

        piece = state.remaining_pieces[action.piece_index]

        # Check that rotation is in allowed_rotations
        rotation = action.rotation_deg
        rotation_allowed = any(
            abs(rotation - r) < 1e-3 for r in piece.allowed_rotations
        )
        if not rotation_allowed:
            return False, (
                f"Rotation {rotation:.1f}° not in allowed rotations "
                f"{piece.allowed_rotations} for piece '{piece.name}'"
            )

        # Check grain alignment
        if not piece.check_grain_alignment(rotation):
            return False, (
                f"Rotation {rotation:.1f}° violates grain direction "
                f"'{piece.grain_direction.value}' for piece '{piece.name}' "
                f"(tolerance: {piece.grain_tolerance_deg:.1f}°)"
            )

        return True, "Valid action"

    def get_max_steps(self) -> int:
        """Return the maximum number of steps per episode for this task.

        Default is 5x the total number of piece instances (with quantities).

        Returns:
            Maximum steps per episode.
        """
        pieces = self.get_pieces()
        total_instances = sum(p.quantity for p in pieces)
        return max(100, total_instances * 5)

    def get_info(self) -> Dict[str, Any]:
        """Return task metadata as a dictionary.

        Returns:
            Dictionary with task name, description, fabric dimensions,
            number of pieces, and total piece area.
        """
        pieces = self.get_pieces()
        fabric_w, fabric_l = self.get_fabric_dimensions()
        total_piece_area = sum(p.area * p.quantity for p in pieces)
        fabric_area = fabric_w * fabric_l

        return {
            "name": self.name,
            "description": self.description,
            "fabric_width": fabric_w,
            "fabric_max_length": fabric_l,
            "n_piece_types": len(pieces),
            "n_piece_instances": sum(p.quantity for p in pieces),
            "total_piece_area_cm2": round(total_piece_area, 2),
            "theoretical_min_utilization": round(total_piece_area / fabric_area, 4),
        }

    def supports_rolling_fabric(self) -> bool:
        """Return whether this task supports rolling fabric extension.

        Returns:
            False by default. Override in IndustrialModeTask.
        """
        return False

    def get_rolling_extension_amount(self) -> float:
        """Return the amount to extend rolling fabric by when needed.

        Returns:
            Extension amount in cm. Default 50.
        """
        return 50.0

    def __repr__(self) -> str:
        """Return a string representation of this task."""
        w, l = self.get_fabric_dimensions()
        return f"{self.__class__.__name__}(name='{self.name}', fabric={w}x{l}cm)"
