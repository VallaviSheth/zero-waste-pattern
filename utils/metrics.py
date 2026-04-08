"""Metrics computation for ZeroWaste-Pattern environment.

Provides functions and classes for measuring environment performance,
including utilization, waste, and episode-level aggregates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from shapely.geometry import Polygon
from shapely.ops import unary_union


def compute_utilization(
    placed_polygons: List[Polygon],
    fabric_width: float,
    fabric_length_used: float,
) -> float:
    """Compute fabric utilization as a fraction of used fabric area.

    Utilization = sum of placed polygon areas / (fabric_width * fabric_length_used).
    Uses fabric_length_used (actual extent of placements) rather than max length.

    Args:
        placed_polygons: List of placed Shapely Polygons.
        fabric_width: Width of the fabric in cm.
        fabric_length_used: Length of fabric consumed (y-extent) in cm.

    Returns:
        Utilization fraction in [0, 1].
    """
    if fabric_width <= 0 or fabric_length_used <= 0:
        return 0.0

    fabric_area = fabric_width * fabric_length_used
    if fabric_area < 1e-6:
        return 0.0

    total_piece_area = sum(p.area for p in placed_polygons)
    utilization = total_piece_area / fabric_area
    return float(np.clip(utilization, 0.0, 1.0))


def compute_waste(
    placed_polygons: List[Polygon],
    fabric_width: float,
    fabric_length_used: float,
) -> float:
    """Compute fabric waste as a fraction of used fabric area not covered by pieces.

    Waste = 1 - utilization.

    Args:
        placed_polygons: List of placed Shapely Polygons.
        fabric_width: Width of the fabric in cm.
        fabric_length_used: Length of fabric consumed (y-extent) in cm.

    Returns:
        Waste fraction in [0, 1].
    """
    return 1.0 - compute_utilization(placed_polygons, fabric_width, fabric_length_used)


def compute_yield(
    placed_polygons: List[Polygon],
    total_piece_area: float,
) -> float:
    """Compute marker yield: fraction of total required piece area that has been placed.

    Args:
        placed_polygons: List of placed Shapely Polygons.
        total_piece_area: Total area of all pieces that need to be placed (cm²).

    Returns:
        Yield fraction in [0, 1].
    """
    if total_piece_area < 1e-6:
        return 0.0
    placed_area = sum(p.area for p in placed_polygons)
    return float(np.clip(placed_area / total_piece_area, 0.0, 1.0))


class EpisodeMetrics(BaseModel):
    """Aggregated metrics for a complete episode.

    Attributes:
        steps: Total number of steps taken.
        placements: Number of successful piece placements.
        utilization_pct: Final fabric utilization percentage (0-100).
        waste_pct: Final waste percentage (0-100).
        marker_yield_pct: Fraction of required piece area placed (0-100).
        total_reward: Cumulative reward for the episode.
        invalid_actions: Number of invalid actions taken.
        fabric_length_used: Fabric length consumed in cm.
        pieces_placed: Number of distinct piece instances placed.
        pieces_total: Total number of piece instances required.
    """

    steps: int = Field(ge=0)
    placements: int = Field(ge=0)
    utilization_pct: float = Field(ge=0.0, le=100.0)
    waste_pct: float = Field(ge=0.0, le=100.0)
    marker_yield_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_reward: float
    invalid_actions: int = Field(ge=0)
    fabric_length_used: float = Field(ge=0.0)
    pieces_placed: int = Field(ge=0)
    pieces_total: int = Field(ge=0)

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """Return a human-readable summary of the metrics."""
        return (
            f"EpisodeMetrics(\n"
            f"  steps={self.steps}, placements={self.placements}, "
            f"pieces={self.pieces_placed}/{self.pieces_total}\n"
            f"  utilization={self.utilization_pct:.1f}%, waste={self.waste_pct:.1f}%, "
            f"yield={self.marker_yield_pct:.1f}%\n"
            f"  total_reward={self.total_reward:.3f}, invalid_actions={self.invalid_actions}\n"
            f"  fabric_length_used={self.fabric_length_used:.1f} cm\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation of the metrics.

        Returns:
            Dictionary with all metric fields.
        """
        return {
            "steps": self.steps,
            "placements": self.placements,
            "utilization_pct": self.utilization_pct,
            "waste_pct": self.waste_pct,
            "marker_yield_pct": self.marker_yield_pct,
            "total_reward": self.total_reward,
            "invalid_actions": self.invalid_actions,
            "fabric_length_used": self.fabric_length_used,
            "pieces_placed": self.pieces_placed,
            "pieces_total": self.pieces_total,
        }


def compute_episode_metrics(history: List[Dict[str, Any]]) -> EpisodeMetrics:
    """Compute episode-level metrics from a history of step records.

    Each step record in history is expected to have keys:
    - "reward": float reward for the step
    - "valid": bool whether the action was valid
    - "placed": bool whether a piece was placed
    - "utilization": float utilization at this step
    - "fabric_length_used": float length used at this step

    Args:
        history: List of step-level dictionaries from episode rollout.

    Returns:
        EpisodeMetrics aggregating the full episode.
    """
    if not history:
        return EpisodeMetrics(
            steps=0,
            placements=0,
            utilization_pct=0.0,
            waste_pct=100.0,
            marker_yield_pct=0.0,
            total_reward=0.0,
            invalid_actions=0,
            fabric_length_used=0.0,
            pieces_placed=0,
            pieces_total=0,
        )

    steps = len(history)
    total_reward = sum(h.get("reward", 0.0) for h in history)
    invalid_actions = sum(1 for h in history if not h.get("valid", True))
    placements = sum(1 for h in history if h.get("placed", False))

    # Use final step values for terminal metrics
    final = history[-1]
    utilization_pct = float(final.get("utilization", 0.0)) * 100.0
    waste_pct = 100.0 - utilization_pct
    fabric_length_used = float(final.get("fabric_length_used", 0.0))

    pieces_placed = int(final.get("pieces_placed", placements))
    pieces_total = int(final.get("pieces_total", 0))
    marker_yield_pct = float(final.get("marker_yield", 0.0)) * 100.0

    return EpisodeMetrics(
        steps=steps,
        placements=placements,
        utilization_pct=float(np.clip(utilization_pct, 0.0, 100.0)),
        waste_pct=float(np.clip(waste_pct, 0.0, 100.0)),
        marker_yield_pct=float(np.clip(marker_yield_pct, 0.0, 100.0)),
        total_reward=total_reward,
        invalid_actions=invalid_actions,
        fabric_length_used=fabric_length_used,
        pieces_placed=pieces_placed,
        pieces_total=pieces_total,
    )


def print_metrics_table(metrics_list: List[EpisodeMetrics], labels: Optional[List[str]] = None) -> None:
    """Print a formatted comparison table of episode metrics.

    Args:
        metrics_list: List of EpisodeMetrics to compare.
        labels: Optional list of label strings for each row.
    """
    if labels is None:
        labels = [f"Episode {i+1}" for i in range(len(metrics_list))]

    # Header
    col_width = 15
    header = (
        f"{'Label':<20} "
        f"{'Utilization':>{col_width}} "
        f"{'Waste':>{col_width}} "
        f"{'Yield':>{col_width}} "
        f"{'Reward':>{col_width}} "
        f"{'Placements':>{col_width}} "
        f"{'Invalid':>{col_width}} "
        f"{'Steps':>{col_width}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for label, m in zip(labels, metrics_list):
        row = (
            f"{label:<20} "
            f"{m.utilization_pct:>{col_width}.2f}% "
            f"{m.waste_pct:>{col_width}.2f}% "
            f"{m.marker_yield_pct:>{col_width}.2f}% "
            f"{m.total_reward:>{col_width}.3f} "
            f"{m.placements:>{col_width}} "
            f"{m.invalid_actions:>{col_width}} "
            f"{m.steps:>{col_width}}"
        )
        print(row)

    print(sep)
