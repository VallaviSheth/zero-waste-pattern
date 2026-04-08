

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field
from shapely.geometry import Polygon

from models.pattern_piece import PatternPiece


class PlacedPiece(BaseModel):
    """A pattern piece that has been successfully placed on the fabric.

    Attributes:
        piece: The original PatternPiece definition.
        x: X coordinate of the bounding-box lower-left corner of the placed polygon.
        y: Y coordinate of the bounding-box lower-left corner of the placed polygon.
        rotation_deg: Rotation angle applied during placement.
        placed_polygon: The actual Shapely polygon at its final position.
        placement_step: The episode step number when this piece was placed.
    """

    piece: PatternPiece
    x: float = Field(ge=0.0)
    y: float = Field(ge=0.0)
    rotation_deg: float
    placed_polygon: Optional[object] = Field(
        default=None, description="Shapely Polygon at final position."
    )
    placement_step: int = Field(default=0, ge=0)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def get_polygon(self) -> Polygon:
        """Return the placed polygon as a Shapely Polygon.

        Returns:
            The Shapely Polygon at its placed position.
        """
        if self.placed_polygon is not None:
            return self.placed_polygon  # type: ignore[return-value]
        # Reconstruct from piece definition if not cached
        from utils.geometry import place_polygon
        return place_polygon(self.piece.polygon, self.x, self.y, self.rotation_deg)


class EnvironmentState(BaseModel):
    """Complete state of the fabric marker making environment.

    Attributes:
        occupancy_grid: Binary H x W numpy array. 1 = occupied, 0 = free.
        remaining_pieces: List of pattern pieces not yet placed.
        placed_pieces: List of PlacedPiece instances placed so far.
        utilization_pct: Current fabric utilization as a percentage (0-100).
        fabric_width: Width of the fabric in cm.
        fabric_length_used: Length of fabric consumed (max y-extent of placements) in cm.
        fabric_max_length: Maximum possible fabric length in cm.
        step_count: Number of steps taken in the current episode.
        episode_done: Whether the episode has ended.
        invalid_action_count: Running count of invalid actions in this episode.
        total_reward: Cumulative reward so far in this episode.
    """

    occupancy_grid: object = Field(description="Binary numpy array H x W.")
    remaining_pieces: List[PatternPiece] = Field(default_factory=list)
    placed_pieces: List[PlacedPiece] = Field(default_factory=list)
    utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    fabric_width: float = Field(gt=0.0)
    fabric_length_used: float = Field(default=0.0, ge=0.0)
    fabric_max_length: float = Field(gt=0.0)
    step_count: int = Field(default=0, ge=0)
    episode_done: bool = Field(default=False)
    invalid_action_count: int = Field(default=0, ge=0)
    total_reward: float = Field(default=0.0)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def get_occupancy_array(self) -> np.ndarray:
        """Return the occupancy grid as a numpy array.

        Returns:
            Binary numpy array of shape (H, W).
        """
        return np.array(self.occupancy_grid, dtype=np.float32)

    def get_placed_polygons(self) -> List[Polygon]:
        """Return all placed polygons as a list.

        Returns:
            List of Shapely Polygons for each placed piece.
        """
        return [pp.get_polygon() for pp in self.placed_pieces]

    @property
    def n_placed(self) -> int:
        """Return the number of pieces placed so far."""
        return len(self.placed_pieces)

    @property
    def n_remaining(self) -> int:
        """Return the number of pieces not yet placed."""
        return len(self.remaining_pieces)

    def copy_state(self) -> "EnvironmentState":
        """Return a shallow copy of this state (for history recording).

        Returns:
            A new EnvironmentState with the same field values.
        """
        grid = np.array(self.occupancy_grid).copy()
        return EnvironmentState(
            occupancy_grid=grid,
            remaining_pieces=list(self.remaining_pieces),
            placed_pieces=list(self.placed_pieces),
            utilization_pct=self.utilization_pct,
            fabric_width=self.fabric_width,
            fabric_length_used=self.fabric_length_used,
            fabric_max_length=self.fabric_max_length,
            step_count=self.step_count,
            episode_done=self.episode_done,
            invalid_action_count=self.invalid_action_count,
            total_reward=self.total_reward,
        )
