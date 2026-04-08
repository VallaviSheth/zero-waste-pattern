"""Typed Reward model for OpenEnv spec compliance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """Reward signal from the ZeroWaste-Pattern environment."""

    total: float = Field(description="Scalar reward for this step.")
    is_valid: bool = Field(default=True, description="Whether the action was valid.")
    reason: str = Field(default="", description="Human-readable reward description.")

    @classmethod
    def from_step_info(cls, reward: float, info: dict) -> "Reward":
        """Construct from scalar reward and info dict."""
        action_result = info.get("action_result", {})
        return cls(
            total=reward,
            is_valid=action_result.get("valid", True),
            reason=action_result.get("reason", ""),
        )
