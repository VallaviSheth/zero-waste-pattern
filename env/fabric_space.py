

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.geometry import (
    check_no_overlap,
    check_within_bounds,
    rasterize_polygon,
)


class FabricSpace:
    """Represents the physical fabric on which pattern pieces are placed.

    Manages a binary occupancy grid for fast collision detection, along with
    the collection of placed Shapely polygons for precise geometric operations.

    Supports optional rolling fabric mode where the fabric length can be
    extended dynamically when placements approach the current end.

    Attributes:
        width: Fabric width in cm.
        max_length: Maximum allowed fabric length in cm (or current in rolling mode).
        cell_size: Size of each square grid cell in cm.
        rolling_fabric: Whether rolling fabric extension is enabled.
        occupancy_grid: Binary numpy array of shape (grid_h, grid_w).
        placed_polygons: List of placed Shapely Polygon objects.
        current_length: Current active length of the fabric in cm.
    """

    def __init__(
        self,
        width: float,
        max_length: float,
        cell_size: float = 2.0,
        rolling_fabric: bool = False,
    ) -> None:
        """Initialize the fabric space.

        Args:
            width: Fabric width in cm.
            max_length: Maximum fabric length in cm. In rolling mode, this
                        is the starting length that can grow.
            cell_size: Size of each square occupancy grid cell in cm.
            rolling_fabric: If True, enable dynamic fabric length extension.
        """
        self.width = width
        self.max_length = max_length
        self.cell_size = cell_size
        self.rolling_fabric = rolling_fabric

        self.grid_w = math.ceil(width / cell_size)
        self.grid_h = math.ceil(max_length / cell_size)

        self.current_length = max_length
        self.placed_polygons: List[Polygon] = []

        # Initialize occupancy grid to all zeros (free)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def reset(self) -> None:
        """Reset the fabric space to its initial empty state.

        Clears all placed polygons and resets the occupancy grid.
        Also resets current_length to max_length (rolling fabric resets too).
        """
        self.current_length = self.max_length
        self.grid_h = math.ceil(self.max_length / self.cell_size)
        self.grid_w = math.ceil(self.width / self.cell_size)
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.placed_polygons = []

    def place_piece(self, polygon: Polygon) -> bool:
        """Place a polygon on the fabric by marking its cells as occupied.

        Rasterizes the polygon onto the occupancy grid and adds it to the
        list of placed polygons.

        Args:
            polygon: The Shapely Polygon at its final placed position.

        Returns:
            True if placement was successful, False if the polygon is empty
            or extends beyond current fabric bounds.
        """
        if polygon.is_empty:
            return False

        # Extend fabric if needed (rolling mode)
        _, _, _, maxy = polygon.bounds
        if maxy > self.current_length:
            if self.rolling_fabric:
                # Calculate needed extension
                extension_needed = maxy - self.current_length
                extension_amount = max(50.0, math.ceil(extension_needed / 50.0) * 50.0)
                self.extend_fabric(extension_amount)
            else:
                return False

        # Rasterize onto occupancy grid
        mask = rasterize_polygon(polygon, self.grid_w, self.grid_h, self.cell_size)
        self.occupancy_grid = np.clip(self.occupancy_grid + mask, 0.0, 1.0)

        # Add to placed polygons list
        self.placed_polygons.append(polygon)
        return True

    def remove_piece(self, polygon: Polygon) -> bool:
        """Remove a previously placed polygon from the fabric.

        Unmarks the cells in the occupancy grid. Used for undo operations.

        Args:
            polygon: The Shapely Polygon to remove (must match a placed polygon).

        Returns:
            True if the polygon was found and removed, False otherwise.
        """
        # Find matching polygon in placed_polygons
        found = False
        for i, placed in enumerate(self.placed_polygons):
            if placed.equals(polygon) or abs(placed.area - polygon.area) < 0.01:
                self.placed_polygons.pop(i)
                found = True
                break

        if not found:
            return False

        # Recompute occupancy grid from remaining polygons
        self.occupancy_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for p in self.placed_polygons:
            mask = rasterize_polygon(p, self.grid_w, self.grid_h, self.cell_size)
            self.occupancy_grid = np.clip(self.occupancy_grid + mask, 0.0, 1.0)

        return True

    def is_valid_placement(
        self,
        polygon: Polygon,
        tolerance: float = 0.1,
    ) -> Tuple[bool, str]:
        """Check whether a polygon can be validly placed on the fabric.

        Checks bounds and overlap constraints.

        Args:
            polygon: The candidate Shapely Polygon at its proposed position.
            tolerance: Minimum intersection area to count as overlap (cm²).

        Returns:
            Tuple of (is_valid: bool, reason: str).
        """
        if polygon.is_empty:
            return False, "Polygon is empty"

        if not polygon.is_valid:
            return False, "Polygon geometry is invalid"

        # Check within fabric width
        minx, miny, maxx, maxy = polygon.bounds

        if minx < -1e-4:
            return False, f"Polygon extends left of fabric (minx={minx:.2f})"

        if maxx > self.width + 1e-4:
            return False, f"Polygon extends right of fabric (maxx={maxx:.2f}, width={self.width:.2f})"

        if miny < -1e-4:
            return False, f"Polygon extends above fabric start (miny={miny:.2f})"

        # Check against current length (extension happens in place_piece)
        if not self.rolling_fabric and maxy > self.current_length + 1e-4:
            return False, (
                f"Polygon extends beyond fabric length "
                f"(maxy={maxy:.2f}, length={self.current_length:.2f})"
            )

        # Check no overlap with existing pieces
        if not check_no_overlap(polygon, self.placed_polygons, tolerance=tolerance):
            return False, "Polygon overlaps an existing placed piece"

        return True, "Valid placement"

    def extend_fabric(self, amount: float) -> None:
        """Extend the fabric length by the given amount.

        In rolling fabric mode, this grows the active length and resizes
        the occupancy grid accordingly.

        Args:
            amount: Amount to extend the fabric by in cm.
        """
        if amount <= 0:
            return

        old_length = self.current_length
        self.current_length += amount
        new_grid_h = math.ceil(self.current_length / self.cell_size)

        if new_grid_h > self.grid_h:
            # Extend the occupancy grid with zeros
            extra_rows = new_grid_h - self.grid_h
            extension = np.zeros((extra_rows, self.grid_w), dtype=np.float32)
            self.occupancy_grid = np.vstack([self.occupancy_grid, extension])
            self.grid_h = new_grid_h

    def get_free_area(self) -> float:
        """Compute the total free (unoccupied) area on the current fabric.

        Returns:
            Free area in cm².
        """
        total_fabric_area = self.width * self.current_length
        occupied_cells = np.sum(self.occupancy_grid)
        occupied_area = occupied_cells * self.cell_size * self.cell_size
        return max(0.0, total_fabric_area - occupied_area)

    def get_bounding_used_length(self) -> float:
        """Compute how far down the fabric has been used.

        Returns the maximum y-coordinate of all placed polygons,
        which represents the actual fabric consumed.

        Returns:
            Maximum y-extent of placed pieces in cm, or 0 if nothing placed.
        """
        if not self.placed_polygons:
            return 0.0

        max_y = 0.0
        for poly in self.placed_polygons:
            _, _, _, maxy = poly.bounds
            max_y = max(max_y, maxy)
        return max_y

    def get_occupancy_for_observation(
        self, target_h: Optional[int] = None, target_w: Optional[int] = None
    ) -> np.ndarray:
        """Return the occupancy grid, optionally resized for observation space.

        Args:
            target_h: Target height in grid cells (rows). If None, use current.
            target_w: Target width in grid cells (columns). If None, use current.

        Returns:
            Float32 numpy array of shape (target_h, target_w).
        """
        grid = self.occupancy_grid.copy()

        if target_h is not None and target_w is not None:
            if target_h != self.grid_h or target_w != self.grid_w:
                # Pad or crop to target size
                result = np.zeros((target_h, target_w), dtype=np.float32)
                copy_h = min(target_h, grid.shape[0])
                copy_w = min(target_w, grid.shape[1])
                result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
                return result

        return grid.astype(np.float32)

    def get_used_cells_ratio(self) -> float:
        """Return the fraction of grid cells that are occupied.

        Returns:
            Ratio of occupied cells to total cells in [0, 1].
        """
        total_cells = self.grid_h * self.grid_w
        if total_cells == 0:
            return 0.0
        return float(np.sum(self.occupancy_grid)) / total_cells

    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Return the current occupancy grid dimensions.

        Returns:
            Tuple of (grid_h, grid_w).
        """
        return (self.grid_h, self.grid_w)

    def has_enough_space(self, min_area: float) -> bool:
        """Check whether there is at least min_area of free space remaining.

        Args:
            min_area: Minimum required free area in cm².

        Returns:
            True if sufficient free space exists.
        """
        return self.get_free_area() >= min_area

    def __repr__(self) -> str:
        """Return a string representation of the FabricSpace."""
        return (
            f"FabricSpace(width={self.width}, length={self.current_length}, "
            f"cell_size={self.cell_size}, "
            f"grid={self.grid_h}x{self.grid_w}, "
            f"pieces={len(self.placed_polygons)}, "
            f"rolling={self.rolling_fabric})"
        )
