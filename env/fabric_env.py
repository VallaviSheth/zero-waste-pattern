

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False
        # Minimal stub for when gymnasium is not installed
        class gym:
            class Env:
                pass
        class spaces:
            class Dict:
                pass

from env.fabric_space import FabricSpace
from models.action import ActionResult, PlacementAction
from models.pattern_piece import GrainDirection, PatternPiece
from models.reward_config import RewardConfig
from models.state import EnvironmentState, PlacedPiece
from tasks.base_task import BaseTask
from utils.geometry import (
    compute_fragmentation,
    place_polygon,
)
from utils.metrics import compute_utilization, compute_yield


# Number of features per piece in the observation
PIECE_FEATURE_DIM = 8
# Features: [norm_width, norm_height, area_ratio, grain_dir_0, grain_dir_1,
#             grain_dir_2, qty_remaining_norm, is_present]

# Maximum number of allowed rotation indices across all tasks
MAX_ROTATIONS = 4


class ZeroWasteFabricEnv(gym.Env):  # type: ignore[misc]
    """ZeroWaste-Pattern: The AI Sustainable Tailor Gymnasium Environment.

    A reinforcement learning environment for garment marker making — the process
    of laying out pattern pieces on fabric to minimize waste. The agent must
    learn to place pieces efficiently while respecting grain direction constraints
    and avoiding overlaps.

    Action Space:
        MultiDiscrete([n_pieces + 1, grid_w, grid_h, max_rotations])
        - dim 0: which piece to place (n_pieces = no-op/skip)
        - dim 1: grid column (x direction)
        - dim 2: grid row (y direction)
        - dim 3: rotation index into piece's allowed_rotations

    Observation Space:
        Dict({
            "occupancy": Box(0, 1, shape=(grid_h, grid_w)),
            "pieces_remaining": Box(0, inf, shape=(max_pieces, PIECE_FEATURE_DIM)),
            "utilization": Box(0, 1, shape=(1,)),
            "fabric_length_used": Box(0, 1, shape=(1,)),
            "step_count": Box(0, 1, shape=(1,)),
        })

    Reward:
        - Successful placement: area_fraction * placement_reward_scale
        - Invalid piece index: invalid_action_penalty
        - Overlap or out-of-bounds: overlap_penalty or out_of_bounds_penalty
        - Grain violation: grain_violation_penalty
        - Per step: step_penalty + fragmentation_penalty
        - Episode end bonus: utilization * completion_bonus_scale

    Attributes:
        task: The BaseTask defining pieces, fabric, and reward config.
        cell_size: Grid cell size in cm.
        max_steps: Maximum steps per episode.
        verbose: Whether to print debug information.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task: BaseTask,
        cell_size: float = 2.0,
        max_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the ZeroWaste Fabric Environment.

        Args:
            task: BaseTask instance defining pieces, fabric, and reward config.
            cell_size: Size of each occupancy grid cell in cm.
            max_steps: Maximum steps per episode. If None, uses task default.
            verbose: Whether to print debug information during episodes.
        """
        super().__init__()

        self.task = task
        self.cell_size = cell_size
        self.verbose = verbose

        # Get task configuration
        self.fabric_width, self.fabric_max_length = task.get_fabric_dimensions()
        self.reward_config: RewardConfig = task.get_reward_config()
        self.max_steps = max_steps if max_steps is not None else task.get_max_steps()

        # Compute grid dimensions
        self.grid_w = math.ceil(self.fabric_width / cell_size)
        self.grid_h = math.ceil(self.fabric_max_length / cell_size)

        # Get pieces and expand by quantity
        self._piece_templates: List[PatternPiece] = task.get_pieces()
        self._all_pieces: List[PatternPiece] = self._expand_pieces(self._piece_templates)
        self.max_pieces = len(self._all_pieces)

        # Total piece area for yield computation
        self._total_piece_area = sum(p.area for p in self._all_pieces)

        # Build action space
        # Piece index: 0..n_pieces-1 for actual pieces, n_pieces for no-op
        n_pieces = self.max_pieces
        if HAS_GYMNASIUM:
            self.action_space = spaces.MultiDiscrete(
                [n_pieces + 1, self.grid_w, self.grid_h, MAX_ROTATIONS]
            )
            # Build observation space
            self.observation_space = spaces.Dict({
                "occupancy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.grid_h, self.grid_w),
                    dtype=np.float32,
                ),
                "pieces_remaining": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.max_pieces, PIECE_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "utilization": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "fabric_length_used": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "step_count": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            })

        # Internal state
        self._fabric_space: Optional[FabricSpace] = None
        self._remaining_pieces: List[PatternPiece] = []
        self._placed_pieces: List[PlacedPiece] = []
        self._step_count: int = 0
        self._invalid_action_count: int = 0
        self._total_reward: float = 0.0
        self._episode_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Gymnasium Interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional dictionary with additional reset options.
                     Supports "randomize_order": bool (shuffle piece order).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize fabric space
        self._fabric_space = FabricSpace(
            width=self.fabric_width,
            max_length=self.fabric_max_length,
            cell_size=self.cell_size,
            rolling_fabric=self.task.supports_rolling_fabric(),
        )

        # Initialize remaining pieces (fresh copies)
        self._remaining_pieces = list(self._all_pieces)

        # Optionally shuffle piece order
        if options and options.get("randomize_order", False):
            np.random.shuffle(self._remaining_pieces)

        self._placed_pieces = []
        self._step_count = 0
        self._invalid_action_count = 0
        self._total_reward = 0.0
        self._episode_history = []

        observation = self._get_observation()
        info = self._get_info()

        if self.verbose:
            print(f"\n[{self.task.name}] Episode reset. {len(self._remaining_pieces)} pieces to place.")

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step of the environment.

        Takes the action, attempts to place the selected piece, computes
        the reward, and returns the new observation.

        Args:
            action: Numpy array of shape (4,) with values
                    [piece_idx, grid_x, grid_y, rotation_idx].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Parse action
        piece_idx = int(action[0])
        grid_x = int(action[1])
        grid_y = int(action[2])
        rotation_idx = int(action[3])

        # No-op action: piece_idx == max_pieces
        if piece_idx >= len(self._remaining_pieces):
            reward = self.reward_config.step_penalty
            self._invalid_action_count += 1
            result = ActionResult(
                valid=False,
                reason="No-op or invalid piece index",
                reward=reward,
            )
        else:
            # Get the selected piece
            piece = self._remaining_pieces[piece_idx]

            # Determine rotation
            rotation_idx = min(rotation_idx, len(piece.allowed_rotations) - 1)
            rotation_deg = piece.allowed_rotations[rotation_idx]

            # Convert grid position to continuous
            x = float(grid_x) * self.cell_size
            y = float(grid_y) * self.cell_size

            placement = PlacementAction(
                piece_index=piece_idx,
                x=x,
                y=y,
                rotation_deg=rotation_deg,
            )

            # Try to execute placement
            result = self._execute_placement(piece, placement)

        # Update reward
        self._total_reward += result.reward

        # Check termination conditions
        terminated = len(self._remaining_pieces) == 0
        truncated = self._step_count >= self.max_steps

        # Add completion bonus if episode is ending
        if terminated or truncated:
            bonus = self._compute_completion_bonus()
            result.reward += bonus
            self._total_reward += bonus
            if self.verbose:
                print(
                    f"[{self.task.name}] Episode ended at step {self._step_count}. "
                    f"Utilization: {self._compute_utilization_pct():.1f}%, "
                    f"Bonus: {bonus:.3f}"
                )

        # Build step history record
        util = self._compute_utilization()
        step_record = {
            "step": self._step_count,
            "reward": result.reward,
            "valid": result.valid,
            "placed": result.valid and piece_idx < len(self._all_pieces),
            "utilization": util,
            "fabric_length_used": self._fabric_space.get_bounding_used_length() if self._fabric_space else 0.0,
            "pieces_placed": len(self._placed_pieces),
            "pieces_total": len(self._all_pieces),
            "marker_yield": compute_yield(
                self._fabric_space.placed_polygons if self._fabric_space else [],
                self._total_piece_area,
            ),
        }
        self._episode_history.append(step_record)

        observation = self._get_observation()
        info = self._get_info()
        info["action_result"] = result.dict()
        info["episode_history"] = self._episode_history

        if self.verbose and result.valid:
            print(
                f"  Step {self._step_count}: Placed '{piece.name if piece_idx < len(self._remaining_pieces) + 1 else 'N/A'}' "
                f"at ({x:.1f}, {y:.1f}) rot={rotation_deg}° reward={result.reward:.3f}"
            )

        return observation, result.reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the current state of the environment.

        Args:
            mode: Render mode. "human" shows the plot, "rgb_array" returns array.

        Returns:
            RGB array if mode="rgb_array", else None.
        """
        if self._fabric_space is None:
            if self.verbose:
                print("Environment not initialized. Call reset() first.")
            return None

        from models.state import EnvironmentState
        from utils.visualization import FabricVisualizer

        state = self._build_env_state()
        fabric_length = self._fabric_space.current_length

        viz = FabricVisualizer(state, self.fabric_width, fabric_length)

        if mode == "human":
            viz.render(show=True)
            return None
        elif mode == "rgb_array":
            import io
            import matplotlib.pyplot as plt
            fig = viz.render(show=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            import matplotlib.image as mpimg
            img = mpimg.imread(buf)
            plt.close(fig)
            return (img[:, :, :3] * 255).astype(np.uint8)

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        import matplotlib.pyplot as plt
        plt.close("all")

    # ------------------------------------------------------------------
    # Action Masking (for MaskablePPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of valid actions for action masking.

        Used with MaskablePPO from sb3-contrib. Returns a flat boolean array
        marking valid (True) and invalid (False) actions.

        The mask has shape matching the flattened action space:
        [(n_pieces+1), grid_w, grid_h, MAX_ROTATIONS]

        Returns:
            Boolean numpy array of shape (n_pieces+1 + grid_w + grid_h + MAX_ROTATIONS,)
            Note: For MultiDiscrete, the mask is concatenated per dimension.
        """
        n_pieces_total = len(self._all_pieces) + 1  # +1 for no-op

        # Piece mask: valid if piece is still remaining
        piece_mask = np.zeros(n_pieces_total, dtype=bool)
        for i, piece in enumerate(self._all_pieces):
            if piece in self._remaining_pieces:
                piece_mask[i] = True
        piece_mask[n_pieces_total - 1] = True  # no-op always valid

        # Grid masks: all positions technically selectable (validity checked at runtime)
        grid_x_mask = np.ones(self.grid_w, dtype=bool)
        grid_y_mask = np.ones(self.grid_h, dtype=bool)

        # Rotation mask: all rotations always available (validity per piece checked at runtime)
        rotation_mask = np.ones(MAX_ROTATIONS, dtype=bool)

        return np.concatenate([piece_mask, grid_x_mask, grid_y_mask, rotation_mask])

    def compute_invalid_mask(self) -> np.ndarray:
        """Compute a binary mask indicating which piece indices are invalid.

        Returns:
            Boolean array of shape (n_pieces + 1,). True = invalid.
        """
        n_total = len(self._all_pieces) + 1
        mask = np.ones(n_total, dtype=bool)  # start all invalid

        for i, piece in enumerate(self._all_pieces):
            if piece in self._remaining_pieces:
                mask[i] = False  # valid

        mask[-1] = False  # no-op always valid (not invalid)
        return mask

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _execute_placement(
        self, piece: PatternPiece, placement: PlacementAction
    ) -> ActionResult:
        """Attempt to execute a piece placement on the fabric.

        Checks all constraints (bounds, overlap, grain) and places the piece
        if valid, computing the appropriate reward.

        Args:
            piece: The PatternPiece to place.
            placement: The PlacementAction with position and rotation.

        Returns:
            ActionResult with validity, reason, and reward.
        """
        rotation_deg = placement.rotation_deg
        x = placement.x
        y = placement.y

        # Step 1: Check grain alignment
        if not piece.check_grain_alignment(rotation_deg):
            self._invalid_action_count += 1
            reward = (
                self.reward_config.grain_violation_penalty
                + self.reward_config.step_penalty
            )
            return ActionResult(
                valid=False,
                reason=f"Grain violation: rotation {rotation_deg}° violates "
                       f"{piece.grain_direction.value} grain for '{piece.name}'",
                reward=reward,
            )

        # Step 2: Place polygon at position
        try:
            placed_poly = place_polygon(piece.polygon, x, y, rotation_deg)
        except Exception as e:
            self._invalid_action_count += 1
            return ActionResult(
                valid=False,
                reason=f"Polygon placement error: {e}",
                reward=self.reward_config.invalid_action_penalty,
            )

        # Step 3: Check bounds
        minx, miny, maxx, maxy = placed_poly.bounds
        if minx < -1e-4 or maxx > self.fabric_width + 1e-4:
            self._invalid_action_count += 1
            reward = (
                self.reward_config.out_of_bounds_penalty
                + self.reward_config.step_penalty
            )
            return ActionResult(
                valid=False,
                reason=f"Out of bounds (x): [{minx:.2f}, {maxx:.2f}] vs width={self.fabric_width}",
                reward=reward,
            )

        if miny < -1e-4:
            self._invalid_action_count += 1
            reward = (
                self.reward_config.out_of_bounds_penalty
                + self.reward_config.step_penalty
            )
            return ActionResult(
                valid=False,
                reason=f"Out of bounds (y below 0): miny={miny:.2f}",
                reward=reward,
            )

        # Check y upper bound (unless rolling fabric)
        if not self.task.supports_rolling_fabric():
            if maxy > self.fabric_max_length + 1e-4:
                self._invalid_action_count += 1
                reward = (
                    self.reward_config.out_of_bounds_penalty
                    + self.reward_config.step_penalty
                )
                return ActionResult(
                    valid=False,
                    reason=f"Out of bounds (y): maxy={maxy:.2f} vs length={self.fabric_max_length}",
                    reward=reward,
                )

        # Step 4: Check overlap
        fabric = self._fabric_space
        is_valid, reason = fabric.is_valid_placement(placed_poly)
        if not is_valid:
            self._invalid_action_count += 1
            reward = (
                self.reward_config.overlap_penalty
                + self.reward_config.step_penalty
            )
            return ActionResult(
                valid=False,
                reason=f"Overlap detected: {reason}",
                reward=reward,
            )

        # Step 5: Place the piece
        success = fabric.place_piece(placed_poly)
        if not success:
            self._invalid_action_count += 1
            return ActionResult(
                valid=False,
                reason="Fabric placement failed (unexpected error)",
                reward=self.reward_config.invalid_action_penalty,
            )

        # Step 6: Record the placement
        piece_idx = self._remaining_pieces.index(piece)
        placed = PlacedPiece(
            piece=piece,
            x=x,
            y=y,
            rotation_deg=rotation_deg,
            placed_polygon=placed_poly,
            placement_step=self._step_count,
        )
        self._placed_pieces.append(placed)
        self._remaining_pieces.pop(piece_idx)

        # Step 7: Compute placement reward
        fabric_area = self.fabric_width * self.fabric_max_length
        area_fraction = piece.area / fabric_area if fabric_area > 0 else 0.0
        placement_reward = area_fraction * self.reward_config.placement_reward_scale

        # Add fragmentation penalty
        frag = compute_fragmentation(
            fabric.placed_polygons,
            self.fabric_width,
            fabric.current_length,
        )
        frag_penalty = -frag * self.reward_config.fragmentation_penalty_scale

        reward = placement_reward + frag_penalty + self.reward_config.step_penalty

        if self.verbose:
            print(
                f"  [Placed] '{piece.name}' area={piece.area:.1f} cm² "
                f"placement_r={placement_reward:.3f} frag_p={frag_penalty:.3f}"
            )

        return ActionResult(
            valid=True,
            reason="Successful placement",
            reward=reward,
            placement_x=x,
            placement_y=y,
            rotation_deg=rotation_deg,
        )

    def _compute_completion_bonus(self) -> float:
        """Compute the end-of-episode completion bonus.

        Returns:
            Completion bonus proportional to final utilization.
        """
        util = self._compute_utilization()
        return util * self.reward_config.completion_bonus_scale

    def _compute_utilization(self) -> float:
        """Compute current fabric utilization fraction.

        Returns:
            Utilization as a fraction in [0, 1].
        """
        if self._fabric_space is None:
            return 0.0
        length_used = max(
            self._fabric_space.get_bounding_used_length(),
            1.0  # avoid division by zero
        )
        return compute_utilization(
            self._fabric_space.placed_polygons,
            self.fabric_width,
            length_used,
        )

    def _compute_utilization_pct(self) -> float:
        """Compute current fabric utilization as a percentage.

        Returns:
            Utilization percentage in [0, 100].
        """
        return self._compute_utilization() * 100.0

    def _get_observation(self) -> Dict[str, Any]:
        """Build the observation dictionary from current environment state.

        Returns:
            Dictionary matching the observation_space specification.
        """
        # Occupancy grid
        if self._fabric_space is not None:
            occupancy = self._fabric_space.get_occupancy_for_observation(
                target_h=self.grid_h, target_w=self.grid_w
            )
        else:
            occupancy = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Piece features
        fabric_area = self.fabric_width * self.fabric_max_length
        pieces_obs = np.zeros((self.max_pieces, PIECE_FEATURE_DIM), dtype=np.float32)

        for i, piece in enumerate(self._all_pieces):
            if piece in self._remaining_pieces:
                w, h = piece.get_natural_width(), piece.get_natural_height()
                pieces_obs[i, 0] = w / self.fabric_width  # normalized width
                pieces_obs[i, 1] = h / self.fabric_max_length  # normalized height
                pieces_obs[i, 2] = piece.area / fabric_area  # area ratio
                # Grain direction one-hot encoding (3 dims for H/V/BIAS, 0 for ANY)
                if piece.grain_direction == GrainDirection.HORIZONTAL:
                    pieces_obs[i, 3] = 1.0
                elif piece.grain_direction == GrainDirection.VERTICAL:
                    pieces_obs[i, 4] = 1.0
                elif piece.grain_direction == GrainDirection.BIAS:
                    pieces_obs[i, 5] = 1.0
                # Quantity remaining (normalized by max 10)
                remaining_qty = sum(1 for p in self._remaining_pieces if p.id.startswith(piece.id.split('_copy')[0]))
                pieces_obs[i, 6] = min(remaining_qty / 10.0, 1.0)
                pieces_obs[i, 7] = 1.0  # is_present

        # Scalar observations
        utilization = np.array([self._compute_utilization()], dtype=np.float32)
        length_used = self._fabric_space.get_bounding_used_length() if self._fabric_space else 0.0
        fabric_length_used_norm = np.array(
            [length_used / self.fabric_max_length], dtype=np.float32
        )
        step_count_norm = np.array(
            [self._step_count / self.max_steps], dtype=np.float32
        )

        return {
            "occupancy": occupancy,
            "pieces_remaining": pieces_obs,
            "utilization": utilization,
            "fabric_length_used": fabric_length_used_norm,
            "step_count": step_count_norm,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Build the info dictionary with diagnostic information.

        Returns:
            Dictionary with episode diagnostics.
        """
        length_used = self._fabric_space.get_bounding_used_length() if self._fabric_space else 0.0
        return {
            "task_name": self.task.name,
            "step_count": self._step_count,
            "pieces_placed": len(self._placed_pieces),
            "pieces_remaining": len(self._remaining_pieces),
            "pieces_total": len(self._all_pieces),
            "utilization_pct": self._compute_utilization_pct(),
            "fabric_length_used": length_used,
            "invalid_action_count": self._invalid_action_count,
            "total_reward": self._total_reward,
        }

    def _build_env_state(self) -> EnvironmentState:
        """Build an EnvironmentState snapshot from current internal state.

        Returns:
            EnvironmentState representing the current environment.
        """
        occupancy = (
            self._fabric_space.occupancy_grid.copy()
            if self._fabric_space else np.zeros((self.grid_h, self.grid_w))
        )
        length_used = (
            self._fabric_space.get_bounding_used_length()
            if self._fabric_space else 0.0
        )
        current_length = (
            self._fabric_space.current_length
            if self._fabric_space else self.fabric_max_length
        )

        return EnvironmentState(
            occupancy_grid=occupancy,
            remaining_pieces=list(self._remaining_pieces),
            placed_pieces=list(self._placed_pieces),
            utilization_pct=self._compute_utilization_pct(),
            fabric_width=self.fabric_width,
            fabric_length_used=length_used,
            fabric_max_length=current_length,
            step_count=self._step_count,
            episode_done=(len(self._remaining_pieces) == 0 or self._step_count >= self.max_steps),
            invalid_action_count=self._invalid_action_count,
            total_reward=self._total_reward,
        )

    @staticmethod
    def _expand_pieces(piece_templates: List[PatternPiece]) -> List[PatternPiece]:
        """Expand piece templates by their quantity into individual instances.

        Args:
            piece_templates: List of PatternPiece instances with quantity fields.

        Returns:
            List of individual piece instances (each with quantity=1).
        """
        expanded = []
        for piece in piece_templates:
            if piece.quantity == 1:
                expanded.append(piece)
            else:
                for q in range(piece.quantity):
                    new_piece = PatternPiece(
                        id=f"{piece.id}_copy{q+1}",
                        name=piece.name,
                        vertices=piece.vertices,
                        grain_direction=piece.grain_direction,
                        grain_tolerance_deg=piece.grain_tolerance_deg,
                        allowed_rotations=piece.allowed_rotations,
                        quantity=1,
                        color=piece.color,
                    )
                    expanded.append(new_piece)
        return expanded

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Return the step-by-step history of the current/last episode.

        Returns:
            List of per-step metric dictionaries.
        """
        return list(self._episode_history)

    def state(self) -> EnvironmentState:
        """Return the current environment state (OpenEnv spec).

        Returns:
            Current EnvironmentState snapshot.
        """
        return self._build_env_state()

    # Backward-compatible alias
    get_current_state = state

    def __repr__(self) -> str:
        """Return a string representation of the environment."""
        return (
            f"ZeroWasteFabricEnv("
            f"task={self.task.name}, "
            f"fabric={self.fabric_width}x{self.fabric_max_length}cm, "
            f"grid={self.grid_h}x{self.grid_w}, "
            f"cell_size={self.cell_size}cm, "
            f"max_pieces={self.max_pieces}, "
            f"max_steps={self.max_steps})"
        )


# ------------------------------------------------------------------
# Gymnasium Registration
# ------------------------------------------------------------------

def _make_basic_env(**kwargs: Any) -> ZeroWasteFabricEnv:
    """Factory function for BasicPacking environment."""
    from tasks.basic_packing import BasicPackingTask
    task = kwargs.pop("task", BasicPackingTask())
    return ZeroWasteFabricEnv(task=task, **kwargs)


def _make_irregular_env(**kwargs: Any) -> ZeroWasteFabricEnv:
    """Factory function for IrregularShapes environment."""
    from tasks.irregular_shapes import IrregularShapesTask
    task = kwargs.pop("task", IrregularShapesTask())
    return ZeroWasteFabricEnv(task=task, **kwargs)


def _make_industrial_env(**kwargs: Any) -> ZeroWasteFabricEnv:
    """Factory function for IndustrialMode environment."""
    from tasks.industrial_mode import IndustrialModeTask
    task = kwargs.pop("task", IndustrialModeTask())
    return ZeroWasteFabricEnv(task=task, **kwargs)


def register_environments() -> None:
    """Register all three ZeroWaste environments with Gymnasium.

    Should be called once at module import or at the start of a training script.
    """
    if not HAS_GYMNASIUM:
        warnings.warn("Gymnasium not installed. Environment registration skipped.")
        return

    try:
        import gymnasium as gym_module
    except ImportError:
        import gym as gym_module  # type: ignore[no-redef]

    env_specs = [
        {
            "id": "ZeroWasteFabric-Basic-v0",
            "entry_point": "env.fabric_env:_make_basic_env",
            "max_episode_steps": 300,
            "kwargs": {},
        },
        {
            "id": "ZeroWasteFabric-Irregular-v0",
            "entry_point": "env.fabric_env:_make_irregular_env",
            "max_episode_steps": 400,
            "kwargs": {},
        },
        {
            "id": "ZeroWasteFabric-Industrial-v0",
            "entry_point": "env.fabric_env:_make_industrial_env",
            "max_episode_steps": 500,
            "kwargs": {},
        },
    ]

    for spec in env_specs:
        env_id = spec["id"]
        try:
            # Check if already registered
            gym_module.spec(env_id)
        except Exception:
            try:
                gym_module.register(
                    id=env_id,
                    entry_point=spec["entry_point"],
                    max_episode_steps=spec["max_episode_steps"],
                    kwargs=spec["kwargs"],
                )
            except Exception as e:
                warnings.warn(f"Failed to register {env_id}: {e}")


# Register environments when module is imported
try:
    register_environments()
except Exception:
    pass
