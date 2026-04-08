"""Typed Observation model extending openenv.core.Observation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from openenv.core import Observation as BaseObservation
from pydantic import Field


class Observation(BaseObservation):
    """Observation from the ZeroWaste-Pattern environment.

    Extends openenv.core.Observation with fabric-specific fields.
    """

    occupancy: List[List[float]] = Field(
        default_factory=list,
        description="Binary occupancy grid (grid_h x grid_w).",
    )
    pieces_remaining: List[List[float]] = Field(
        default_factory=list,
        description="Per-piece feature matrix (max_pieces x 8).",
    )
    utilization: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fabric utilization fraction.",
    )
    fabric_length_used: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Normalized fabric length consumed.",
    )
    current_step: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Normalized step count.",
    )

    @classmethod
    def from_gym_obs(cls, obs: dict, done: bool = False,
                     reward: float = 0.0) -> "Observation":
        """Construct from raw gymnasium observation dict."""
        occ = obs["occupancy"]
        pr = obs["pieces_remaining"]
        return cls(
            occupancy=occ.tolist() if isinstance(occ, np.ndarray) else occ,
            pieces_remaining=pr.tolist() if isinstance(pr, np.ndarray) else pr,
            utilization=float(obs["utilization"].flat[0]),
            fabric_length_used=float(obs["fabric_length_used"].flat[0]),
            current_step=float(obs["step_count"].flat[0]),
            done=done,
            reward=reward,
        )
