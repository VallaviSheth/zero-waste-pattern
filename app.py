"""FastAPI server for ZeroWaste-Pattern OpenEnv environment.

Serves the environment as an HTTP API for Hugging Face Spaces deployment.
Tagged as an `openenv` space.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.fabric_env import ZeroWasteFabricEnv
from models.observation import Observation
from models.reward import Reward
from tasks.basic_packing import BasicPackingTask
from tasks.industrial_mode import IndustrialModeTask
from tasks.irregular_shapes import IrregularShapesTask

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ZeroWaste-Pattern OpenEnv",
    description=(
        "RL environment for garment marker making — arrange pattern pieces "
        "on fabric to minimize waste. OpenEnv-compliant API."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

TASKS = {
    "basic": BasicPackingTask,
    "irregular": IrregularShapesTask,
    "industrial": IndustrialModeTask,
}

sessions: Dict[str, ZeroWasteFabricEnv] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = Field(
        default="basic",
        description="Task id: 'basic', 'irregular', or 'industrial'.",
    )
    seed: Optional[int] = Field(default=None, description="Random seed.")


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation
    info: Dict[str, Any]


class StepRequest(BaseModel):
    session_id: str
    action: List[int] = Field(
        description="Action array [piece_idx, grid_x, grid_y, rotation_idx].",
        min_length=4,
        max_length=4,
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    session_id: str
    utilization_pct: float
    pieces_placed: int
    pieces_remaining: int
    step_count: int
    fabric_length_used: float
    total_reward: float
    grade: float


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    fabric_width: float
    fabric_max_length: float
    n_piece_types: int
    n_piece_instances: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check and environment metadata."""
    return {
        "name": "ZeroWaste-Pattern",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": list(TASKS.keys()),
        "description": (
            "RL environment for garment marker making. "
            "Arrange pattern pieces on fabric to minimize waste."
        ),
    }


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    """List all available tasks with metadata."""
    result = []
    difficulty_map = {"basic": "easy", "irregular": "medium", "industrial": "hard"}
    for task_id, TaskCls in TASKS.items():
        task = TaskCls()
        info = task.get_info()
        result.append(TaskInfo(
            id=task_id,
            name=info["name"],
            description=info["description"],
            difficulty=difficulty_map[task_id],
            fabric_width=info["fabric_width"],
            fabric_max_length=info["fabric_max_length"],
            n_piece_types=info["n_piece_types"],
            n_piece_instances=info["n_piece_instances"],
        ))
    return result


@app.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest = ResetRequest()):
    """Reset the environment and start a new episode.

    Returns a session_id for subsequent step() and state() calls.
    """
    if req.task not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'. Choose from: {list(TASKS.keys())}",
        )

    task = TASKS[req.task]()
    env = ZeroWasteFabricEnv(task=task, max_steps=300)
    obs_raw, info = env.reset(seed=req.seed)

    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = env

    # Prune old sessions (keep max 100)
    if len(sessions) > 100:
        oldest = list(sessions.keys())[0]
        del sessions[oldest]

    obs = Observation.from_gym_obs(obs_raw)

    # Filter info to JSON-serializable values
    safe_info = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}

    return ResetResponse(
        session_id=session_id,
        observation=obs,
        info=safe_info,
    )


@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    """Execute one step in the environment."""
    env = sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found.")

    action = np.array(req.action, dtype=np.int64)
    obs_raw, reward, terminated, truncated, info = env.step(action)
    obs = Observation.from_gym_obs(obs_raw)

    # Filter info
    safe_info = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}

    # Clean up session if episode ended
    if terminated or truncated:
        # Keep session alive briefly so state() can be called
        pass

    return StepResponse(
        observation=obs,
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=safe_info,
    )


@app.get("/state/{session_id}", response_model=StateResponse)
def get_state(session_id: str):
    """Return the current environment state (OpenEnv spec)."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    state = env.state()
    grade = env.task.grade(state)

    return StateResponse(
        session_id=session_id,
        utilization_pct=round(state.utilization_pct, 2),
        pieces_placed=state.n_placed,
        pieces_remaining=state.n_remaining,
        step_count=state.step_count,
        fabric_length_used=round(state.fabric_length_used, 2),
        total_reward=round(state.total_reward, 4),
        grade=round(grade, 4),
    )


@app.get("/grade/{session_id}")
def grade_episode(session_id: str):
    """Grade the current episode (0.0-1.0)."""
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    state = env.state()
    grade = env.task.grade(state)

    return {
        "session_id": session_id,
        "task": env.task.name,
        "grade": round(grade, 4),
        "utilization_pct": round(state.utilization_pct, 2),
        "pieces_placed": state.n_placed,
        "pieces_total": state.n_placed + state.n_remaining,
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session to free resources."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
