# ZeroWaste-Pattern: The AI Sustainable Tailor

An OpenEnv-compliant reinforcement learning environment for **garment marker making** -- the industrial process of arranging pattern pieces on fabric rolls to minimize material waste.

In the garment industry, marker making wastes 15-25% of fabric. This environment trains AI agents to act as digital tailors, optimally placing irregular polygon-shaped pattern pieces while respecting grain direction constraints and manufacturing tolerances.

---

## Environment Overview

| Property | Value |
|---|---|
| Domain | Garment manufacturing / combinatorial optimization |
| Observation | Dict: occupancy grid + piece features + utilization scalars |
| Action | MultiDiscrete: [piece_idx, grid_x, grid_y, rotation_idx] |
| Reward | Shaped: placement bonus + violation penalties + completion bonus |
| Tasks | 3 (easy / medium / hard) |
| Graders | Per-task, deterministic, 0.0-1.0 scoring |

---

## Tasks

### Task 1: BasicPacking (Easy)
- **Pieces:** 12 rectangular shapes, no grain constraints
- **Fabric:** 150 x 300 cm
- **Rotations:** 0, 90, 180, 270 degrees (all allowed)
- **Target:** 75% utilization
- **Grading:** 60% utilization score + 40% placement ratio

### Task 2: IrregularShapes (Medium)
- **Pieces:** 10 real shirt polygon shapes (bodice, sleeves, collar, cuffs, yoke)
- **Fabric:** 150 x 400 cm
- **Constraints:** Per-piece grain direction (VERTICAL / HORIZONTAL)
- **Target:** 70% utilization
- **Grading:** 50% utilization + 30% placement ratio + 20% efficiency

### Task 3: IndustrialMode (Hard)
- **Pieces:** 10 industrial shirt pieces with strict 2.5-degree grain tolerance
- **Fabric:** 150 x 500 cm (rolling -- extends by 50cm when needed)
- **Constraints:** Most pieces limited to 0/180 degree rotation only
- **Target:** 80% utilization
- **Grading:** 40% utilization + 25% placement + 20% efficiency + 15% fabric economy

---

## Action Space

```
MultiDiscrete([n_pieces + 1, grid_w, grid_h, 4])
```

| Dimension | Range | Description |
|---|---|---|
| `piece_idx` | 0 to n_pieces | Which piece to place (n_pieces = no-op) |
| `grid_x` | 0 to grid_w-1 | X position on fabric grid (x = grid_x * cell_size) |
| `grid_y` | 0 to grid_h-1 | Y position on fabric grid (y = grid_y * cell_size) |
| `rotation_idx` | 0 to 3 | Rotation: 0=0deg, 1=90deg, 2=180deg, 3=270deg |

---

## Observation Space

```python
Dict({
    "occupancy":        Box(0, 1, shape=(grid_h, grid_w)),       # binary occupancy grid
    "pieces_remaining": Box(0, inf, shape=(max_pieces, 8)),      # per-piece features
    "utilization":      Box(0, 1, shape=(1,)),                   # current utilization fraction
    "fabric_length_used": Box(0, 1, shape=(1,)),                 # normalized length consumed
    "step_count":       Box(0, 1, shape=(1,)),                   # normalized step count
})
```

**Piece features (8 dims):** normalized_width, normalized_height, area_ratio, grain_horizontal, grain_vertical, grain_bias, qty_remaining_norm, is_present

---

## Reward Function

| Component | Signal | Description |
|---|---|---|
| Placement reward | `+(area/fabric_area) * scale` | Positive reward proportional to placed piece area |
| Overlap penalty | `-1.0` | Piece overlaps existing placement |
| Out-of-bounds penalty | `-0.8` | Piece extends beyond fabric edges |
| Grain violation penalty | `-0.6` | Rotation violates grain direction constraint |
| Fragmentation penalty | `-frag * 0.05` | Penalizes fragmented free space |
| Step penalty | `-0.01` | Small per-step cost for efficiency |
| Completion bonus | `+utilization * scale` | End-of-episode bonus for high utilization |

The reward provides dense signal throughout the episode -- not just sparse end-of-episode feedback.

---

## Setup

### Local Installation

```bash
git clone <repo-url>
cd Hackathon
pip install -r requirements.txt
```

### Docker

```bash
docker build -t zerowaste-pattern .
docker run -p 7860:7860 zerowaste-pattern
```

### Hugging Face Spaces

Deploy as a Docker Space tagged with `openenv`. The app serves a FastAPI server on port 7860.

---

## Usage

### Python (Gymnasium API)

```python
from env.fabric_env import ZeroWasteFabricEnv
from tasks.basic_packing import BasicPackingTask

env = ZeroWasteFabricEnv(task=BasicPackingTask())
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# OpenEnv spec
state = env.state()
grade = env.task.grade(state)
print(f"Grade: {grade:.4f}, Utilization: {state.utilization_pct:.1f}%")
```

### Registered Gymnasium Environments

```python
import gymnasium as gym
env = gym.make("ZeroWasteFabric-Basic-v0")
env = gym.make("ZeroWasteFabric-Irregular-v0")
env = gym.make("ZeroWasteFabric-Industrial-v0")
```

### HTTP API (FastAPI)

```bash
# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Reset
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task": "basic", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "action": [0, 10, 5, 0]}'

# Get state
curl http://localhost:7860/state/<id>

# Grade
curl http://localhost:7860/grade/<id>
```

### Training with PPO (stable-baselines3)

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("ZeroWasteFabric-Basic-v0")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
```

---

## Baseline Inference

### Random Agent (no API key needed)

```bash
python baseline_inference.py --random-only
```

### LLM Agent (OpenAI API)

```bash
export OPENAI_API_KEY=sk-...
python baseline_inference.py --model gpt-4o-mini --episodes 1
```

### Other Scripts

```bash
python random_agent_rollout.py     # Visual demo with metrics
python heuristic_baseline.py       # Greedy bottom-left packer comparison
python example_usage.py            # Full training loop walkthrough
```

---

## Baseline Scores

| Task | Random Agent | Greedy Heuristic | Target |
|---|---|---|---|
| BasicPacking (easy) | ~0.25 grade | ~0.85 grade | 0.90+ |
| IrregularShapes (medium) | ~0.15 grade | ~0.70 grade | 0.80+ |
| IndustrialMode (hard) | ~0.10 grade | ~0.60 grade | 0.75+ |

Scores are on a 0.0-1.0 scale using the task-specific grader.

---

## Project Structure

```
.
├── app.py                    # FastAPI server (HF Space)
├── Dockerfile                # Container build
├── openenv.yaml              # OpenEnv specification
├── requirements.txt          # Python dependencies
├── baseline_inference.py     # OpenAI API baseline script
├── example_usage.py          # Training loop examples
├── random_agent_rollout.py   # Random agent demo
├── heuristic_baseline.py     # Greedy packer comparison
├── env/
│   ├── fabric_env.py         # Main Gymnasium environment
│   └── fabric_space.py       # Fabric grid + Shapely collision
├── models/
│   ├── pattern_piece.py      # PatternPiece, GrainDirection
│   ├── action.py             # PlacementAction, DiscreteAction
│   ├── state.py              # EnvironmentState, PlacedPiece
│   ├── observation.py        # Typed Observation (OpenEnv)
│   ├── reward.py             # Typed Reward (OpenEnv)
│   └── reward_config.py      # RewardConfig presets
├── tasks/
│   ├── base_task.py          # Abstract BaseTask with grade()
│   ├── basic_packing.py      # Easy: rectangles
│   ├── irregular_shapes.py   # Medium: shirt polygons
│   └── industrial_mode.py    # Hard: strict constraints
└── utils/
    ├── geometry.py            # Shapely helpers
    ├── visualization.py       # Matplotlib renderer
    ├── dataset.py             # Pattern dataset generator
    └── metrics.py             # Utilization/waste metrics
```

---

## OpenEnv Spec Compliance

- `reset()` -> returns initial observation
- `step(action)` -> returns (observation, reward, terminated, truncated, info)
- `state()` -> returns current EnvironmentState
- `task.grade(state)` -> returns 0.0-1.0 score
- Typed Pydantic models: `Observation`, `Reward`, `PlacementAction`, `EnvironmentState`
- `openenv.yaml` with metadata, tasks, schemas, and grading criteria

---

## License

MIT
