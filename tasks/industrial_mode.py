"""Industrial mode task for ZeroWaste-Pattern environment.

Task 3: Strict grain tolerances, fixed rotations, rolling fabric simulation,
and multiple quantities per piece. Mimics real garment industry constraints.
"""

from __future__ import annotations

from typing import List, Tuple

from models.pattern_piece import GrainDirection, PatternPiece
from models.reward_config import RewardConfig
from models.state import EnvironmentState
from tasks.base_task import BaseTask


class IndustrialModeTask(BaseTask):
    """Industrial garment marker making task with strict manufacturing constraints.

    Simulates real-world garment cutting floor conditions:
    - Very strict grain tolerance (2.5 degrees) for fabric structure integrity
    - Only 0 and 180 degree rotations allowed (no sideways cutting)
    - Rolling fabric: fabric extends by 50 cm when needed
    - Multiple piece quantities (e.g., sleeve x2, cuff x2)
    - High penalties for any constraint violations

    Fabric: 150 cm wide x 500 cm initial length (can grow)
    Pieces: Industrial shirt cut with strict grain and quantity requirements
    Constraints:
        - Grain tolerance: 2.5 degrees (strict)
        - Most pieces: rotations [0, 180] only
        - Small pieces: rotations [0, 90, 180, 270]
        - Rolling fabric: extends by 50 cm when piece won't fit
    Reward: Strict configuration with high violation penalties
    """

    @property
    def name(self) -> str:
        """Return the task name."""
        return "IndustrialMode"

    @property
    def description(self) -> str:
        """Return the task description."""
        return (
            "Industrial garment marker making with strict manufacturing constraints. "
            "Strict grain tolerance (2.5 deg), fixed rotations, rolling fabric simulation, "
            "and multiple quantities per piece. High penalties for constraint violations. "
            "Goal: minimize fabric waste while satisfying all industrial cutting standards."
        )

    def grade(self, state: EnvironmentState) -> float:
        """Grade agent performance on industrial mode (hard task).

        Scoring breakdown:
        - 40% weight: utilization percentage (target 80%+)
        - 25% weight: piece placement ratio (all pieces must be placed)
        - 20% weight: efficiency (low invalid action rate)
        - 15% weight: fabric economy (use less fabric length = better)

        Args:
            state: Final EnvironmentState at episode end.

        Returns:
            Score in [0.0, 1.0].
        """
        util = state.utilization_pct / 100.0
        util_score = min(util / 0.80, 1.0)

        total_pieces = state.n_placed + state.n_remaining
        placement_ratio = state.n_placed / max(total_pieces, 1)

        total_actions = state.step_count if state.step_count > 0 else 1
        invalid_rate = state.invalid_action_count / total_actions
        efficiency = max(1.0 - invalid_rate, 0.0)

        # Fabric economy: how much of max length was used (less = better)
        length_ratio = state.fabric_length_used / state.fabric_max_length
        fabric_economy = max(1.0 - length_ratio, 0.0) if state.n_placed > 0 else 0.0

        score = (
            0.40 * util_score
            + 0.25 * placement_ratio
            + 0.20 * efficiency
            + 0.15 * fabric_economy
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def get_pieces(self) -> List[PatternPiece]:
        """Return the industrial pattern piece set with strict constraints.

        All major pieces have strict 2.5-degree grain tolerance and are
        limited to 0/180-degree rotations. Small accessories allow all rotations.

        Returns:
            List of PatternPiece instances with industrial constraints.
        """
        pieces = [
            # ---- Front Bodice (1 per garment) ----
            PatternPiece(
                id="ind_front_bodice",
                name="Front Bodice",
                vertices=[
                    (0.0, 0.0),
                    (40.0, 0.0),
                    (42.0, 10.0),
                    (44.0, 30.0),
                    (40.0, 55.0),
                    (32.0, 60.0),
                    (24.0, 58.0),
                    (18.0, 52.0),
                    (12.0, 52.0),
                    (6.0, 58.0),
                    (0.0, 60.0),
                    (-2.0, 55.0),
                    (-4.0, 40.0),
                    (-3.0, 20.0),
                    (0.0, 10.0),
                ],
                grain_direction=GrainDirection.VERTICAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=1,
                color="#E74C3C",
            ),
            # ---- Back Bodice (1 per garment) ----
            PatternPiece(
                id="ind_back_bodice",
                name="Back Bodice",
                vertices=[
                    (0.0, 0.0),
                    (42.0, 0.0),
                    (44.0, 10.0),
                    (46.0, 30.0),
                    (43.0, 55.0),
                    (34.0, 62.0),
                    (25.0, 63.0),
                    (10.0, 62.0),
                    (0.0, 62.0),
                    (-2.0, 58.0),
                    (-4.0, 40.0),
                    (-3.0, 20.0),
                    (0.0, 10.0),
                ],
                grain_direction=GrainDirection.VERTICAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=1,
                color="#3498DB",
            ),
            # ---- Sleeve (2 per garment - left and right) ----
            PatternPiece(
                id="ind_sleeve",
                name="Sleeve",
                vertices=[
                    (0.0, 0.0),
                    (20.0, 0.0),
                    (22.0, 5.0),
                    (25.0, 15.0),
                    (26.0, 30.0),
                    (24.0, 45.0),
                    (20.0, 55.0),
                    (16.0, 60.0),
                    (10.0, 63.0),
                    (4.0, 60.0),
                    (0.0, 55.0),
                    (-4.0, 45.0),
                    (-5.0, 30.0),
                    (-3.0, 15.0),
                    (-1.0, 5.0),
                ],
                grain_direction=GrainDirection.VERTICAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=2,
                color="#2ECC71",
            ),
            # ---- Collar Upper (1 per garment) ----
            PatternPiece(
                id="ind_collar_upper",
                name="Collar Upper",
                vertices=[
                    (0.0, 0.0),
                    (30.0, 0.0),
                    (32.0, 3.0),
                    (33.0, 7.0),
                    (30.0, 12.0),
                    (20.0, 13.0),
                    (15.0, 12.0),
                    (10.0, 13.0),
                    (0.0, 12.0),
                    (-3.0, 7.0),
                    (-2.0, 3.0),
                ],
                grain_direction=GrainDirection.HORIZONTAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=1,
                color="#F39C12",
            ),
            # ---- Collar Under (1 per garment) ----
            PatternPiece(
                id="ind_collar_under",
                name="Collar Under",
                vertices=[
                    (0.0, 0.0),
                    (30.0, 0.0),
                    (32.0, 3.0),
                    (33.0, 7.0),
                    (30.0, 12.0),
                    (20.0, 13.0),
                    (15.0, 12.0),
                    (10.0, 13.0),
                    (0.0, 12.0),
                    (-3.0, 7.0),
                    (-2.0, 3.0),
                ],
                grain_direction=GrainDirection.HORIZONTAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=1,
                color="#9B59B6",
            ),
            # ---- Collar Stand (2 per garment - inner/outer) ----
            PatternPiece(
                id="ind_collar_stand",
                name="Collar Stand",
                vertices=[
                    (0.0, 0.0),
                    (32.0, 0.0),
                    (33.0, 2.0),
                    (33.5, 5.0),
                    (33.0, 8.0),
                    (32.0, 8.0),
                    (0.0, 8.0),
                    (-1.0, 5.0),
                    (-0.5, 2.0),
                ],
                grain_direction=GrainDirection.HORIZONTAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=2,
                color="#1ABC9C",
            ),
            # ---- Cuff (2 per garment) ----
            PatternPiece(
                id="ind_cuff",
                name="Cuff",
                vertices=[
                    (0.0, 0.0),
                    (25.0, 0.0),
                    (26.0, 2.0),
                    (26.0, 8.0),
                    (25.0, 10.0),
                    (0.0, 10.0),
                    (-1.0, 8.0),
                    (-1.0, 2.0),
                ],
                grain_direction=GrainDirection.HORIZONTAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=2,
                color="#E67E22",
            ),
            # ---- Front Button Band (2 per garment) ----
            PatternPiece(
                id="ind_front_panel",
                name="Front Button Band",
                vertices=[
                    (0.0, 0.0),
                    (5.0, 0.0),
                    (5.5, 5.0),
                    (6.0, 20.0),
                    (5.5, 40.0),
                    (5.0, 55.0),
                    (4.5, 60.0),
                    (0.0, 60.0),
                    (-0.5, 55.0),
                    (-1.0, 40.0),
                    (-0.5, 20.0),
                    (0.0, 5.0),
                ],
                grain_direction=GrainDirection.VERTICAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=2,
                color="#2980B9",
            ),
            # ---- Back Yoke (2 per garment - outer/inner) ----
            PatternPiece(
                id="ind_back_yoke",
                name="Back Yoke",
                vertices=[
                    (0.0, 0.0),
                    (42.0, 0.0),
                    (44.0, 3.0),
                    (44.0, 15.0),
                    (42.0, 18.0),
                    (30.0, 20.0),
                    (15.0, 20.0),
                    (0.0, 18.0),
                    (-2.0, 15.0),
                    (-2.0, 3.0),
                ],
                grain_direction=GrainDirection.HORIZONTAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 180.0],
                quantity=2,
                color="#27AE60",
            ),
            # ---- Pocket (1 per garment) ----
            PatternPiece(
                id="ind_pocket",
                name="Chest Pocket",
                vertices=[
                    (0.0, 0.0),
                    (12.0, 0.0),
                    (13.0, 1.0),
                    (13.0, 13.0),
                    (12.5, 15.0),
                    (6.5, 16.0),
                    (6.0, 16.0),
                    (0.0, 15.0),
                    (-0.5, 13.0),
                    (-0.5, 1.0),
                ],
                grain_direction=GrainDirection.VERTICAL,
                grain_tolerance_deg=2.5,
                allowed_rotations=[0.0, 90.0, 180.0, 270.0],
                quantity=1,
                color="#8E44AD",
            ),
        ]
        return pieces

    def get_fabric_dimensions(self) -> Tuple[float, float]:
        """Return the initial fabric dimensions for industrial mode.

        The fabric starts at 150 x 500 cm but can extend in rolling mode.

        Returns:
            Tuple of (150.0, 500.0) for 150 cm wide x 500 cm initial length.
        """
        return (150.0, 500.0)

    def get_reward_config(self) -> RewardConfig:
        """Return the strict industrial reward configuration.

        Returns:
            Strict RewardConfig with high penalties for constraint violations.
        """
        return RewardConfig.strict()

    def supports_rolling_fabric(self) -> bool:
        """Industrial mode supports rolling fabric extension.

        Returns:
            True - this task supports extending the fabric during an episode.
        """
        return True

    def get_rolling_extension_amount(self) -> float:
        """Return the rolling extension amount of 50 cm.

        Returns:
            50.0 cm - the amount to extend fabric when more space is needed.
        """
        return 50.0

    def get_max_steps(self) -> int:
        """Return 500 max steps for the industrial task.

        Returns:
            Maximum steps per episode.
        """
        return 500
