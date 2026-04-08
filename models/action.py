
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PlacementAction(BaseModel):
    """A continuous placement action specifying exact position and rotation.

    Attributes:
        piece_index: Index into the list of remaining pieces to place.
        x: X coordinate of the piece's bounding-box lower-left corner (cm).
        y: Y coordinate of the piece's bounding-box lower-left corner (cm).
        rotation_deg: Rotation angle in degrees to apply before placement.
    """

    piece_index: int = Field(ge=0, description="Index of the piece to place.")
    x: float = Field(ge=0.0, description="X placement coordinate in cm.")
    y: float = Field(ge=0.0, description="Y placement coordinate in cm.")
    rotation_deg: float = Field(default=0.0, description="Rotation in degrees.")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class DiscreteAction(BaseModel):
    """A discrete grid-indexed action for use with standard RL algorithms.

    Attributes:
        piece_idx: Index into the remaining pieces list (or n_pieces for no-op).
        grid_x: Column index in the fabric occupancy grid.
        grid_y: Row index in the fabric occupancy grid.
        rotation_idx: Index into the piece's allowed_rotations list.
    """

    piece_idx: int = Field(ge=0, description="Piece index (n_pieces = no-op).")
    grid_x: int = Field(ge=0, description="Grid column index.")
    grid_y: int = Field(ge=0, description="Grid row index.")
    rotation_idx: int = Field(ge=0, description="Index into allowed_rotations.")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def to_continuous(self, cell_size: float) -> PlacementAction:
        """Convert discrete grid action to a continuous placement action.

        The top-left corner of the selected cell maps to (x, y) coordinates.

        Args:
            cell_size: Size of each grid cell in cm.

        Returns:
            Equivalent PlacementAction with real-valued coordinates.
        """
        return PlacementAction(
            piece_index=self.piece_idx,
            x=float(self.grid_x) * cell_size,
            y=float(self.grid_y) * cell_size,
            rotation_deg=0.0,  # rotation_deg will be looked up from piece's allowed_rotations
        )


class ActionResult(BaseModel):
    """Result of attempting to execute a placement action.

    Attributes:
        valid: Whether the placement was successfully executed.
        reason: Human-readable description of success or failure reason.
        reward: Scalar reward signal for this action.
        placement_x: Actual x coordinate used (after snapping, if any).
        placement_y: Actual y coordinate used.
        rotation_deg: Actual rotation applied.
    """

    valid: bool = Field(description="True if placement was successfully executed.")
    reason: str = Field(description="Description of outcome.")
    reward: float = Field(description="Reward signal for this action.")
    placement_x: Optional[float] = Field(default=None, description="Actual x placement.")
    placement_y: Optional[float] = Field(default=None, description="Actual y placement.")
    rotation_deg: Optional[float] = Field(default=None, description="Actual rotation applied.")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
