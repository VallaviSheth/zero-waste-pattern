"""OpenEnv server for ZeroWaste-Pattern.

Uses openenv.core.create_fastapi_app to serve the environment.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from openenv.core import create_fastapi_app

from env.openenv_wrapper import ZeroWasteEnvironment
from models.observation import Observation
from models.openenv_action import FabricAction

app = create_fastapi_app(
    env=lambda: ZeroWasteEnvironment(task_name="basic"),
    action_cls=FabricAction,
    observation_cls=Observation,
)

def main():
    """Entry point for `server` console script."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
