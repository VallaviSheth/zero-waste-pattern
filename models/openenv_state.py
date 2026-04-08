"""Typed State model extending openenv.core.State."""

from __future__ import annotations

from openenv.core import State as BaseState
from pydantic import Field


class FabricState(BaseState):
    """State of the ZeroWaste-Pattern environment.

    Extends openenv.core.State with fabric-specific fields.
    """

    utilization_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Fabric utilization percentage.",
    )
    pieces_placed: int = Field(
        default=0, ge=0,
        description="Number of pieces placed.",
    )
    pieces_remaining: int = Field(
        default=0, ge=0,
        description="Number of pieces not yet placed.",
    )
    fabric_length_used: float = Field(
        default=0.0, ge=0.0,
        description="Fabric length consumed in cm.",
    )
    fabric_width: float = Field(
        default=150.0, gt=0.0,
        description="Fabric width in cm.",
    )
    fabric_max_length: float = Field(
        default=300.0, gt=0.0,
        description="Max fabric length in cm.",
    )
    invalid_action_count: int = Field(
        default=0, ge=0,
        description="Number of invalid actions so far.",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward.",
    )
    grade: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Current grader score (0.0-1.0).",
    )
    task_name: str = Field(
        default="",
        description="Name of the current task.",
    )
