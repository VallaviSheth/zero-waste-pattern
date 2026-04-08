"""Typed Reward model for OpenEnv spec compliance.

Provides a strongly-typed Pydantic model for the reward signal
returned by the environment at each step.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """Typed reward signal from the ZeroWaste-Pattern environment.

    Decomposes the scalar reward into its constituent components for
    interpretability and debugging.

    Attributes:
        total: Scalar reward value for this step.
        placement_reward: Positive reward for successful piece placement.
        fragmentation_penalty: Penalty for fragmenting free space.
        step_penalty: Small constant per-step penalty.
        violation_penalty: Penalty for invalid actions (overlap/OOB/grain).
        completion_bonus: End-of-episode bonus proportional to utilization.
        is_valid: Whether the action was valid this step.
        reason: Human-readable description of the reward.
    """

    total: float = Field(description="Scalar reward for this step.")
    placement_reward: float = Field(default=0.0, description="Area-based placement reward.")
    fragmentation_penalty: float = Field(default=0.0, description="Fragmentation penalty.")
    step_penalty: float = Field(default=0.0, description="Per-step penalty.")
    violation_penalty: float = Field(default=0.0, description="Invalid action penalty.")
    completion_bonus: float = Field(default=0.0, description="End-of-episode bonus.")
    is_valid: bool = Field(default=True, description="Whether the action was valid.")
    reason: str = Field(default="", description="Human-readable reward description.")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_step_info(cls, reward: float, info: dict) -> "Reward":
        """Construct a Reward from the scalar reward and info dict.

        Args:
            reward: Scalar reward from env.step().
            info: Info dictionary from env.step().

        Returns:
            Typed Reward instance.
        """
        action_result = info.get("action_result", {})
        return cls(
            total=reward,
            is_valid=action_result.get("valid", True),
            reason=action_result.get("reason", ""),
        )
