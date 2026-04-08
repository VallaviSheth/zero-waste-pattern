

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """Configuration for all reward signals in the environment.

    Reward components:
    - Placement reward: proportional to placed area relative to total fabric area.
    - Invalid action penalty: for choosing a piece index out of range.
    - Overlap penalty: for attempting to place on already-occupied cells.
    - Out-of-bounds penalty: for attempting to place outside the fabric.
    - Grain violation penalty: for placing a piece against its grain constraint.
    - Completion bonus: proportional to final utilization, awarded at episode end.
    - Fragmentation penalty: penalizes highly fragmented free space.
    - Step penalty: small negative reward each step to encourage efficiency.

    Attributes:
        placement_reward_scale: Multiplier for per-placement area-based reward.
        invalid_action_penalty: Reward for choosing an invalid piece index.
        overlap_penalty: Reward for attempting a placement that overlaps.
        out_of_bounds_penalty: Reward for attempting out-of-bounds placement.
        grain_violation_penalty: Reward for violating grain direction constraint.
        completion_bonus_scale: Multiplier for end-of-episode utilization bonus.
        fragmentation_penalty_scale: Scale of fragmentation penalty per step.
        step_penalty: Constant per-step penalty.
    """

    placement_reward_scale: float = Field(
        default=10.0,
        description="Reward = (piece_area / total_fabric_area) * scale at each placement.",
    )
    invalid_action_penalty: float = Field(
        default=-0.5,
        description="Penalty for selecting an invalid piece index.",
    )
    overlap_penalty: float = Field(
        default=-1.0,
        description="Penalty for attempting a placement that overlaps existing pieces.",
    )
    out_of_bounds_penalty: float = Field(
        default=-0.8,
        description="Penalty for attempting to place outside fabric bounds.",
    )
    grain_violation_penalty: float = Field(
        default=-0.6,
        description="Penalty for violating grain direction constraint.",
    )
    completion_bonus_scale: float = Field(
        default=5.0,
        description="Bonus = utilization_fraction * scale at episode end.",
    )
    fragmentation_penalty_scale: float = Field(
        default=0.05,
        description="Penalty proportional to fragmentation metric per step.",
    )
    step_penalty: float = Field(
        default=-0.01,
        description="Small constant penalty per step to encourage efficiency.",
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @classmethod
    def lenient(cls) -> "RewardConfig":
        """Return a lenient reward configuration suitable for easy tasks.

        Returns:
            RewardConfig with reduced penalties and higher placement rewards.
        """
        return cls(
            placement_reward_scale=15.0,
            invalid_action_penalty=-0.1,
            overlap_penalty=-0.3,
            out_of_bounds_penalty=-0.2,
            grain_violation_penalty=-0.1,
            completion_bonus_scale=8.0,
            fragmentation_penalty_scale=0.01,
            step_penalty=-0.005,
        )

    @classmethod
    def strict(cls) -> "RewardConfig":
        """Return a strict reward configuration for industrial tasks.

        Returns:
            RewardConfig with high penalties for constraint violations.
        """
        return cls(
            placement_reward_scale=10.0,
            invalid_action_penalty=-1.0,
            overlap_penalty=-2.0,
            out_of_bounds_penalty=-1.5,
            grain_violation_penalty=-2.0,
            completion_bonus_scale=10.0,
            fragmentation_penalty_scale=0.1,
            step_penalty=-0.02,
        )
