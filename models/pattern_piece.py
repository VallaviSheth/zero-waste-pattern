

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon


class GrainDirection(str, Enum):
    """Fabric grain direction for a pattern piece.

    The fabric grain runs along the X-axis (0 degrees) by convention.
    VERTICAL: grain line runs vertically on the piece (90 deg in natural orientation)
    HORIZONTAL: grain line runs horizontally on the piece (0 deg in natural orientation)
    BIAS: grain line is at 45 degrees
    ANY: no grain constraint
    """

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    BIAS = "bias"
    ANY = "any"


class PatternPiece(BaseModel):
    """A single garment pattern piece with geometric and manufacturing constraints.

    Attributes:
        id: Unique identifier for the piece.
        name: Human-readable name (e.g., "front_bodice").
        vertices: List of (x, y) coordinates defining the polygon outline in cm.
        grain_direction: Required fabric grain orientation for this piece.
        grain_tolerance_deg: Allowed angular deviation from perfect grain alignment.
        allowed_rotations: List of rotation angles in degrees permitted for this piece.
        quantity: Number of times this piece must be cut.
        color: Optional hex color string for visualization.
    """

    id: str
    name: str
    vertices: List[Tuple[float, float]]
    grain_direction: GrainDirection = GrainDirection.ANY
    grain_tolerance_deg: float = Field(default=5.0, ge=0.0, le=90.0)
    allowed_rotations: List[float] = Field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    quantity: int = Field(default=1, ge=1)
    color: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @validator("vertices")
    def validate_vertices(cls, v: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Ensure the polygon has at least 3 vertices."""
        if len(v) < 3:
            raise ValueError("A pattern piece polygon must have at least 3 vertices.")
        return v

    @property
    def polygon(self) -> Polygon:
        """Return the Shapely Polygon for this piece in its natural orientation."""
        return Polygon(self.vertices)

    @property
    def area(self) -> float:
        """Return the area of the piece in square cm."""
        return self.polygon.area

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) bounding box in natural orientation."""
        return self.polygon.bounds  # type: ignore[return-value]

    def rotated_polygon(self, angle_deg: float) -> Polygon:
        """Return the piece polygon rotated by angle_deg degrees around its centroid.

        Args:
            angle_deg: Rotation angle in degrees (counter-clockwise).

        Returns:
            Rotated Shapely Polygon.
        """
        poly = self.polygon
        rotated = shapely_rotate(poly, angle_deg, origin="centroid", use_radians=False)
        return rotated  # type: ignore[return-value]

    def check_grain_alignment(self, rotation_deg: float) -> bool:
        """Check whether placing a piece at the given rotation satisfies grain constraints.

        Convention: The fabric grain (warp/selvage direction) runs along the Y-axis
        (the fabric length direction). This is the standard garment industry convention
        where fabric unrolls along Y and the selvage runs parallel to Y.

        For VERTICAL grain: grain line runs vertically on the piece in natural orientation
            (parallel to piece center-front line, i.e., at 90 deg). After rotation by r:
            effective grain angle = 90 + r. For alignment with Y-axis (90 deg), check
            if abs((effective_angle % 180) - 90) <= tolerance.
            Valid rotations: 0 deg and 180 deg (piece upright or flipped).

        For HORIZONTAL grain: grain line runs horizontally on the piece in natural
            orientation (at 0 deg, across the piece). After rotation by r:
            effective angle = r. For alignment with X-axis (0/180 deg), check
            if (effective_angle % 180) <= tolerance or >= (180 - tolerance).
            Valid rotations: 0 deg and 180 deg.

        For BIAS grain: grain line is at 45 deg on the piece. After rotation by r:
            effective angle = 45 + r. Valid if near 45 deg mod 180.

        For ANY: always valid.

        Args:
            rotation_deg: The rotation applied to the piece when placing it.

        Returns:
            True if the placement satisfies the grain constraint.
        """
        if self.grain_direction == GrainDirection.ANY:
            return True

        tol = self.grain_tolerance_deg

        if self.grain_direction == GrainDirection.VERTICAL:
            # Natural grain line is at 90 deg (runs top-to-bottom on piece).
            # After rotation by r, grain angle = (90 + r).
            # Valid if grain runs along Y-axis (90 deg) after placement:
            # i.e., abs((90 + r) mod 180 - 90) <= tol.
            effective_angle = (90.0 + rotation_deg) % 180.0
            return abs(effective_angle - 90.0) <= tol

        elif self.grain_direction == GrainDirection.HORIZONTAL:
            # Natural grain line is at 0 deg (runs left-to-right on piece).
            # After rotation by r, grain angle = r.
            # Valid if grain runs along X-axis (0/180 deg):
            # i.e., (r mod 180) <= tol or >= (180 - tol).
            effective_angle = rotation_deg % 180.0
            return effective_angle <= tol or effective_angle >= (180.0 - tol)

        elif self.grain_direction == GrainDirection.BIAS:
            # Natural grain line is at 45 deg.
            # After rotation by r, grain angle = (45 + r).
            # Valid if near 45 or 225 (mod 180 = 45).
            effective_angle = (45.0 + rotation_deg) % 180.0
            return abs(effective_angle - 45.0) <= tol or abs(effective_angle - 135.0) <= tol

        return True

    def get_natural_width(self) -> float:
        """Return the width (x-extent) of the piece in natural orientation."""
        minx, _, maxx, _ = self.bounding_box
        return maxx - minx

    def get_natural_height(self) -> float:
        """Return the height (y-extent) of the piece in natural orientation."""
        _, miny, _, maxy = self.bounding_box
        return maxy - miny

    def __hash__(self) -> int:
        """Hash based on piece ID for use in sets and dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on piece ID."""
        if isinstance(other, PatternPiece):
            return self.id == other.id
        return False
