"""OpenEnv-compliant wrapper around ZeroWasteFabricEnv.

Implements the openenv.core.Environment interface.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from openenv.core import Environment

from env.fabric_env import ZeroWasteFabricEnv
from models.observation import Observation
from models.openenv_action import FabricAction
from models.openenv_state import FabricState
from tasks.basic_packing import BasicPackingTask
from tasks.irregular_shapes import IrregularShapesTask
from tasks.industrial_mode import IndustrialModeTask


TASK_MAP = {
    "basic": BasicPackingTask,
    "irregular": IrregularShapesTask,
    "industrial": IndustrialModeTask,
}


class ZeroWasteEnvironment(Environment[FabricAction, Observation, FabricState]):
    """OpenEnv-compliant environment for garment marker making.

    Wraps the Gymnasium-based ZeroWasteFabricEnv to conform to the
    openenv.core.Environment interface.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "basic", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        TaskCls = TASK_MAP.get(task_name, BasicPackingTask)
        self._task = TaskCls()
        self._env = ZeroWasteFabricEnv(task=self._task, max_steps=300)
        self._task_name = task_name
        self._last_obs_raw: Optional[dict] = None
        self._last_reward: float = 0.0
        self._done: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment and return initial observation."""
        self._reset_rubric()
        obs_raw, info = self._env.reset(seed=seed)
        self._last_obs_raw = obs_raw
        self._last_reward = 0.0
        self._done = False
        return Observation.from_gym_obs(obs_raw, done=False, reward=0.0)

    def step(
        self,
        action: FabricAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Take a step in the environment."""
        gym_action = np.array([
            action.piece_idx,
            action.grid_x,
            action.grid_y,
            action.rotation_idx,
        ], dtype=np.int64)

        obs_raw, reward, terminated, truncated, info = self._env.step(gym_action)
        self._last_obs_raw = obs_raw
        self._done = terminated or truncated

        # Apply rubric if available
        obs = Observation.from_gym_obs(obs_raw, done=self._done, reward=reward)
        rubric_reward = self._apply_rubric(action, obs)
        if rubric_reward != 0.0:
            obs.reward = rubric_reward

        self._last_reward = float(obs.reward) if obs.reward is not None else 0.0
        return obs

    @property
    def state(self) -> FabricState:
        """Get the current environment state."""
        env_state = self._env.state()
        grade = self._task.grade(env_state)
        return FabricState(
            step_count=env_state.step_count,
            utilization_pct=round(env_state.utilization_pct, 2),
            pieces_placed=env_state.n_placed,
            pieces_remaining=env_state.n_remaining,
            fabric_length_used=round(env_state.fabric_length_used, 2),
            fabric_width=env_state.fabric_width,
            fabric_max_length=env_state.fabric_max_length,
            invalid_action_count=env_state.invalid_action_count,
            total_reward=round(env_state.total_reward, 4),
            grade=round(grade, 4),
            task_name=self._task.name,
        )

    def get_metadata(self):
        """Return environment metadata."""
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ZeroWaste-Pattern",
            description=(
                "RL environment for garment marker making. "
                "Arrange pattern pieces on fabric to minimize waste."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        """Clean up resources."""
        self._env.close()
