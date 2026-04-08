"""Geometry utility functions for ZeroWaste-Pattern environment.

Provides Shapely-based helpers for rotating, translating, and validating
polygon placements on the fabric grid.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from models.pattern_piece import GrainDirection


def rotate_polygon(
    polygon: Polygon,
    angle_deg: float,
    origin: Union[str, Tuple[float, float]] = "centroid",
) -> Polygon:
    """Rotate a Shapely polygon by the given angle.

    Args:
        polygon: The polygon to rotate.
        angle_deg: Counter-clockwise rotation angle in degrees.
        origin: Center of rotation. Can be "centroid", "center", or a (x, y) tuple.

    Returns:
        Rotated Shapely Polygon.
    """
    rotated = shapely_rotate(polygon, angle_deg, origin=origin, use_radians=False)
    return rotated  # type: ignore[return-value]


def translate_polygon(polygon: Polygon, dx: float, dy: float) -> Polygon:
    """Translate a Shapely polygon by (dx, dy).

    Args:
        polygon: The polygon to translate.
        dx: Horizontal displacement in cm.
        dy: Vertical displacement in cm.

    Returns:
        Translated Shapely Polygon.
    """
    translated = shapely_translate(polygon, xoff=dx, yoff=dy)
    return translated  # type: ignore[return-value]


def place_polygon(
    polygon: Polygon,
    x: float,
    y: float,
    angle_deg: float,
) -> Polygon:
    """Rotate a polygon around its centroid, then place so bounding-box min-corner is at (x, y).

    This is the canonical placement operation:
    1. Rotate around centroid.
    2. Translate so the bounding box lower-left corner is at (x, y).

    Args:
        polygon: The original polygon in its natural orientation.
        x: Target x coordinate of bounding box lower-left corner.
        y: Target y coordinate of bounding box lower-left corner.
        angle_deg: Rotation to apply around the centroid.

    Returns:
        Placed Shapely Polygon at the specified position.
    """
    # Step 1: Rotate around centroid
    rotated = shapely_rotate(polygon, angle_deg, origin="centroid", use_radians=False)

    # Step 2: Compute shift needed so bbox min-corner lands at (x, y)
    minx, miny, _, _ = rotated.bounds
    dx = x - minx
    dy = y - miny

    # Step 3: Translate
    placed = shapely_translate(rotated, xoff=dx, yoff=dy)
    return placed  # type: ignore[return-value]


def check_within_bounds(
    polygon: Polygon,
    fabric_w: float,
    fabric_l: float,
) -> bool:
    """Check whether a polygon is fully within the fabric rectangle.

    The fabric occupies [0, fabric_w] x [0, fabric_l].

    Args:
        polygon: The polygon to check (already in placed position).
        fabric_w: Fabric width in cm.
        fabric_l: Fabric length in cm.

    Returns:
        True if the polygon is fully within bounds.
    """
    minx, miny, maxx, maxy = polygon.bounds
    return (
        minx >= -1e-6
        and miny >= -1e-6
        and maxx <= fabric_w + 1e-6
        and maxy <= fabric_l + 1e-6
    )


def check_no_overlap(
    polygon: Polygon,
    placed_polygons: List[Polygon],
    tolerance: float = 0.1,
) -> bool:
    """Check that a polygon does not overlap any already-placed polygon.

    Args:
        polygon: The candidate polygon to check.
        placed_polygons: List of already-placed polygons.
        tolerance: Minimum allowable intersection area (cm²). Small overlaps
                   below this threshold are ignored to handle floating point errors.

    Returns:
        True if there are no significant overlaps.
    """
    for placed in placed_polygons:
        if polygon.intersects(placed):
            intersection_area = polygon.intersection(placed).area
            if intersection_area > tolerance:
                return False
    return True


def rasterize_polygon(
    polygon: Polygon,
    grid_w: int,
    grid_h: int,
    cell_size: float,
) -> np.ndarray:
    """Rasterize a polygon onto a binary occupancy grid.

    A cell is marked as occupied if the polygon's intersection with the cell
    has area greater than half the cell area.

    Args:
        polygon: The polygon to rasterize (in cm coordinates).
        grid_w: Number of grid columns (x direction).
        grid_h: Number of grid rows (y direction).
        cell_size: Size of each square grid cell in cm.

    Returns:
        Binary numpy array of shape (grid_h, grid_w). 1 = occupied, 0 = free.
    """
    mask = np.zeros((grid_h, grid_w), dtype=np.float32)

    if polygon.is_empty:
        return mask

    # Get bounding box of the polygon to limit iteration
    minx, miny, maxx, maxy = polygon.bounds
    col_start = max(0, int(math.floor(minx / cell_size)))
    col_end = min(grid_w, int(math.ceil(maxx / cell_size)))
    row_start = max(0, int(math.floor(miny / cell_size)))
    row_end = min(grid_h, int(math.ceil(maxy / cell_size)))

    threshold = 0.25 * cell_size * cell_size  # 25% of cell area

    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            # Create cell polygon
            cx0 = col * cell_size
            cy0 = row * cell_size
            cx1 = cx0 + cell_size
            cy1 = cy0 + cell_size
            cell = Polygon([(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)])
            try:
                if polygon.intersects(cell):
                    inter_area = polygon.intersection(cell).area
                    if inter_area >= threshold:
                        mask[row, col] = 1.0
            except Exception:
                pass

    return mask


def compute_fragmentation(
    placed_polygons: List[Polygon],
    fabric_w: float,
    fabric_l: float,
) -> float:
    """Compute a fragmentation metric for the free space on the fabric.

    Uses the ratio of actual free area to convex hull area of free space.
    A higher value means more fragmented free space (harder to use efficiently).

    Args:
        placed_polygons: List of placed polygons.
        fabric_w: Fabric width in cm.
        fabric_l: Fabric length in cm.

    Returns:
        Fragmentation score in [0, 1]. 0 = no fragmentation, 1 = fully fragmented.
    """
    if fabric_w <= 0 or fabric_l <= 0:
        return 0.0

    fabric_rect = Polygon(
        [(0, 0), (fabric_w, 0), (fabric_w, fabric_l), (0, fabric_l)]
    )

    if not placed_polygons:
        return 0.0

    try:
        combined = unary_union(placed_polygons)
        free_space = fabric_rect.difference(combined)

        if free_space.is_empty:
            return 0.0

        free_area = free_space.area
        convex_hull_area = free_space.convex_hull.area

        if convex_hull_area < 1e-6:
            return 0.0

        # Fragmentation = 1 - (actual free area / convex hull of free area)
        # Higher = more fragmented
        fragmentation = 1.0 - (free_area / convex_hull_area)
        return float(np.clip(fragmentation, 0.0, 1.0))

    except Exception:
        return 0.0


def get_grain_angle(grain_direction: GrainDirection) -> float:
    """Return the base angle in degrees of the grain line for a direction type.

    The fabric grain runs along the X-axis (0 degrees).
    These angles represent the natural orientation of the grain line on the piece.

    Args:
        grain_direction: The grain direction enum value.

    Returns:
        Base angle in degrees.
    """
    angles = {
        GrainDirection.HORIZONTAL: 0.0,
        GrainDirection.VERTICAL: 90.0,
        GrainDirection.BIAS: 45.0,
        GrainDirection.ANY: 0.0,
    }
    return angles.get(grain_direction, 0.0)


def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """Compute the area of a polygon given its vertices using the shoelace formula.

    Args:
        vertices: List of (x, y) coordinate pairs.

    Returns:
        Area of the polygon in square units.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def get_bounding_box_after_rotation(
    polygon: Polygon, angle_deg: float
) -> Tuple[float, float, float, float]:
    """Get the bounding box of a polygon after rotating it around its centroid.

    Args:
        polygon: The polygon to rotate.
        angle_deg: Rotation angle in degrees.

    Returns:
        Tuple (width, height) of the bounding box after rotation.
    """
    rotated = shapely_rotate(polygon, angle_deg, origin="centroid", use_radians=False)
    minx, miny, maxx, maxy = rotated.bounds
    return (maxx - minx, maxy - miny, minx, miny)
