"""Typed Observation model for OpenEnv spec compliance.

Provides a strongly-typed Pydantic model wrapping the raw observation
dictionary returned by the environment.
"""

from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Typed observation from the ZeroWaste-Pattern environment.

    All fields correspond to entries in the gymnasium observation_space Dict.

    Attributes:
        occupancy: Binary 2D grid (grid_h x grid_w). 1=occupied, 0=free.
        pieces_remaining: Feature matrix (max_pieces x 8) for remaining pieces.
            Features: [norm_w, norm_h, area_ratio, grain_h, grain_v, grain_bias,
                       qty_remaining_norm, is_present].
        utilization: Current fabric utilization fraction in [0, 1].
        fabric_length_used: Normalized fabric length consumed in [0, 1].
        step_count: Normalized step count in [0, 1].
    """

    occupancy: List[List[float]] = Field(
        description="Binary occupancy grid (grid_h x grid_w)."
    )
    pieces_remaining: List[List[float]] = Field(
        description="Per-piece feature matrix (max_pieces x 8)."
    )
    utilization: float = Field(
        ge=0.0, le=1.0,
        description="Fabric utilization fraction.",
    )
    fabric_length_used: float = Field(
        ge=0.0, le=1.0,
        description="Normalized fabric length consumed.",
    )
    step_count: float = Field(
        ge=0.0, le=1.0,
        description="Normalized step count.",
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_gym_obs(cls, obs: dict) -> "Observation":
        """Construct an Observation from the raw gymnasium observation dict.

        Args:
            obs: Dictionary returned by env.reset() or env.step().

        Returns:
            Typed Observation instance.
        """
        occ = obs["occupancy"]
        pr = obs["pieces_remaining"]
        return cls(
            occupancy=occ.tolist() if isinstance(occ, np.ndarray) else occ,
            pieces_remaining=pr.tolist() if isinstance(pr, np.ndarray) else pr,
            utilization=float(obs["utilization"][0]) if hasattr(obs["utilization"], "__getitem__") else float(obs["utilization"]),
            fabric_length_used=float(obs["fabric_length_used"][0]) if hasattr(obs["fabric_length_used"], "__getitem__") else float(obs["fabric_length_used"]),
            step_count=float(obs["step_count"][0]) if hasattr(obs["step_count"], "__getitem__") else float(obs["step_count"]),
        )

    def to_gym_obs(self) -> dict:
        """Convert back to a raw gymnasium observation dict.

        Returns:
            Dictionary compatible with the environment's observation_space.
        """
        return {
            "occupancy": np.array(self.occupancy, dtype=np.float32),
            "pieces_remaining": np.array(self.pieces_remaining, dtype=np.float32),
            "utilization": np.array([self.utilization], dtype=np.float32),
            "fabric_length_used": np.array([self.fabric_length_used], dtype=np.float32),
            "step_count": np.array([self.step_count], dtype=np.float32),
        }
