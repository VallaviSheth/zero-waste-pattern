"""Utility package for ZeroWaste-Pattern environment."""

from utils.geometry import (
    rotate_polygon,
    translate_polygon,
    place_polygon,
    check_within_bounds,
    check_no_overlap,
    rasterize_polygon,
    compute_fragmentation,
    get_grain_angle,
)
from utils.metrics import (
    compute_utilization,
    compute_waste,
    compute_yield,
    EpisodeMetrics,
    compute_episode_metrics,
)
from utils.dataset import PatternDataset
from utils.visualization import FabricVisualizer

__all__ = [
    "rotate_polygon",
    "translate_polygon",
    "place_polygon",
    "check_within_bounds",
    "check_no_overlap",
    "rasterize_polygon",
    "compute_fragmentation",
    "get_grain_angle",
    "compute_utilization",
    "compute_waste",
    "compute_yield",
    "EpisodeMetrics",
    "compute_episode_metrics",
    "PatternDataset",
    "FabricVisualizer",
]
