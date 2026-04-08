"""Typed Action model extending openenv.core.Action."""

from __future__ import annotations

from openenv.core import Action as BaseAction
from pydantic import Field


class FabricAction(BaseAction):
    """Action for the ZeroWaste-Pattern environment.

    Extends openenv.core.Action with placement-specific fields.
    """

    piece_idx: int = Field(
        default=0, ge=0,
        description="Index of piece to place (n_pieces = no-op).",
    )
    grid_x: int = Field(
        default=0, ge=0,
        description="Grid column for placement.",
    )
    grid_y: int = Field(
        default=0, ge=0,
        description="Grid row for placement.",
    )
    rotation_idx: int = Field(
        default=0, ge=0, le=3,
        description="Rotation index: 0=0deg, 1=90deg, 2=180deg, 3=270deg.",
    )
