# Multi-Floor Maze: Reinforcement Learning trong mê cung nhiều tầng

Agent phải đi từ điểm bắt đầu đến đích trong một mê cung được sinh ngẫu nhiên. Đường đi có thể qua nhiều tầng, cửa cần chìa khóa, bẫy và kẻ địch. Mỗi lượt, agent nhìn thấy vùng lân cận và chọn một hành động; mục tiêu là **đến đích trước khi hết máu hoặc hết số bước**.

Đây là bài toán học tăng cường với các episode hữu hạn. Mỗi episode tạo một map mới; agent tương tác với môi trường qua API Gymnasium `reset()` và `step(action)`. Curriculum gồm 9 cấp độ, từ map một tầng không có địch đến map ba tầng có khóa, bẫy và nhiều loại địch.

## Môi trường và agent

Môi trường chính là [`MazeEnv`](multi_floor_maze/mfm/env.py). Map dạng lưới được tạo bằng BSP, nối các phòng bằng hành lang, đặt cầu thang giữa các tầng và kiểm tra đường đi. Agent có máu, đạn, thể lực và số chìa khóa đang giữ. Các ô có thể chứa tường, bẫy, cửa, chìa khóa, gói hồi máu/đạn, cầu thang hoặc đích. Kẻ địch gồm patrol, chaser và sniper.

Ở mỗi bước, môi trường nhận hành động, cập nhật vị trí và tương tác, di chuyển kẻ địch/đạn, rồi trả về `(observation, reward, terminated, truncated, info)`. Episode kết thúc khi agent tới đích hoặc chết (`terminated`), hoặc khi đạt giới hạn số bước (`truncated`). Cấu hình 9 level nằm trong `CURRICULUM` ở cùng file; có thể truyền `config` riêng để tạo map tùy chỉnh.

Ví dụ khởi tạo một episode (chạy từ thư mục `multi_floor_maze`):

```python
from mfm.env import CURRICULUM, MazeEnv

env = MazeEnv(config=CURRICULUM[1])
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(3)  # đi sang phải
```

### Observation space

Actor nhận một `Dict` gồm hai phần:

| Thành phần | Kích thước mặc định | Nội dung |
|---|---:|---|
| `visual` | `16 × 9 × 9` | Vùng 9 × 9 quanh agent, mã hóa theo kênh: tường, bẫy, cửa, cầu thang, đích, vật phẩm, agent, ba loại địch, đạn, tia laser và ô đã thăm. |
| `vector` | `15` | Trạng thái đã chuẩn hóa: máu, đạn, thể lực, chìa khóa, tiếng ồn, tầng hiện tại, hướng tới mục tiêu trung gian, hướng tầng đích, hướng di chuyển gần nhất, thời gian còn lại, mức khám phá, trạng thái đứng trên cầu thang và khoảng cách tới địch gần nhất. |

`visual` chỉ chứa vùng cục bộ; `vector` cung cấp thêm tín hiệu dẫn hướng. Bán kính quan sát mặc định là 4 ô và có thể đổi bằng `view_radius`.

### Action space

Actor chọn **một trong 10 hành động rời rạc** (`Discrete(10)`):

| ID | Hành động |
|---:|---|
| `0–3` | Đi lên, xuống, trái, phải. |
| `4–7` | Bắn lên, xuống, trái, phải; tốn 1 viên đạn. |
| `8` | Lướt tối đa 2 ô theo hướng gần nhất; tốn thể lực. |
| `9` | Dùng cầu thang khi đang đứng trên ô cầu thang. |

Di chuyển vào cửa sẽ tự mở nếu agent có chìa khóa. Bắn và lướt tạo tiếng ồn, có thể khiến địch chú ý.

### Reward

Reward mặc định khuyến khích hoàn thành nhiệm vụ và khám phá: tới đích `+25`, nhặt chìa `+2.5`, mở cửa `+3.5`, thăm ô mới `+0.05`, tiến gần mục tiêu trung gian tối đa `+0.10` mỗi bước. Môi trường phạt mỗi bước `−0.02`, va tường/hành động không hợp lệ `−0.12`, trúng đòn `−2`, dẫm bẫy `−2.5` và chết `−18`. Ngoài ra còn có reward cho hạ địch, dùng cầu thang lần đầu và nhặt vật phẩm. Có thể điều chỉnh các giá trị bằng `config["rewards"]`.

## Thuật toán và cách train

Dự án dùng **PPO** và **A2C** từ Stable-Baselines3 với `MultiInputPolicy`: CNN xử lý `visual`, MLP xử lý `vector`, sau đó ghép đặc trưng để actor chọn hành động và critic ước lượng giá trị. Training dùng 4 môi trường `DummyVecEnv`, chuẩn hóa reward bằng `VecNormalize`, và lần lượt train qua 9 level trong `CURRICULUM`. Sau mỗi level, script đánh giá tỉ lệ thắng; nếu dưới ngưỡng, train thêm một đợt rồi chuyển sang level tiếp theo.

Các script chính:

| Script | Mục đích |
|---|---|
| [`train.py`](multi_floor_maze/train.py) | Train PPO cơ bản trên CPU. |
| [`train_a2c.py`](multi_floor_maze/train_a2c.py) | Train A2C cơ bản trên CPU. |
| [`train_experiments.py`](multi_floor_maze/train_experiments.py) | So sánh 5 cấu hình: PPO, A2C, PPO với `γ=0.95`, mạng nhỏ hơn, và PPO với entropy coefficient `0.02`. Mặc định đặt device là `cuda`. |
| [`evaluate_experiments.py`](multi_floor_maze/evaluate_experiments.py) | Đánh giá các model và random baseline trên nhiều map; xuất JSON vào `outputs/evals/`. |

## Cài đặt và chạy

Chạy các lệnh từ thư mục `multi_floor_maze` (khuyến nghị Python 3.10+):

```bash
cd multi_floor_maze
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python train.py                                  # PPO cơ bản
python train_a2c.py                              # A2C cơ bản
python train_experiments.py --only exp2_a2c      # một thí nghiệm
python evaluate_experiments.py --n-maps 100       # đánh giá model đã train
```

Checkpoint và kết quả được lưu trong `multi_floor_maze/outputs/` (thư mục này không được đưa lên Git). Training đầy đủ qua 9 level tốn nhiều thời gian; `train_experiments.py` cần PyTorch có CUDA nếu giữ nguyên cấu hình device.

Để xem agent chơi trên map tùy chỉnh, cài thêm `Pillow` cho chức năng GIF rồi chạy:

```bash
pip install pillow
streamlit run app_custom_map.py
```

Ứng dụng Streamlit cho phép chọn kích thước, số tầng, bẫy, địch, khóa và model đã train. Ngoài ra có [`play_custom_map.py`](multi_floor_maze/play_custom_map.py) cho CLI, [`generate_gifs.py`](multi_floor_maze/generate_gifs.py) để xuất demo, và [`render_maps.py`](multi_floor_maze/render_maps.py) để vẽ map. Các notebook trong `multi_floor_maze/` phục vụ nghiên cứu sinh map và curriculum.
=======
# Agent-RF Multifloor Maze Game

A **multi-floor maze** environment for **Reinforcement Learning**, where an agent navigates procedurally generated maps, moves between floors, avoids traps, unlocks doors with keys, fights enemies, and reaches the goal. The project includes a Gymnasium environment, a PPO/A2C training pipeline with a 9-level curriculum, algorithm comparison experiments, evaluation tools, and a Streamlit UI to watch the agent play.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Game Environment (`MazeEnv`)](#game-environment-mazeenv)
- [Curriculum — 9 Difficulty Levels](#curriculum--9-difficulty-levels)
- [Procedural Map Generation](#procedural-map-generation)
- [RL Training](#rl-training)
- [5 Comparison Experiments](#5-comparison-experiments)
- [Evaluation & Results Export](#evaluation--results-export)
- [Demos, GIFs & Web UI](#demos-gifs--web-ui)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Outputs](#outputs)
- [Hardest Level Demo](#hardest-level-demo)
- [License](#license)

---

## Overview

This is a **grid-based maze game** with multiple floors. Each episode:

1. A map is **randomly generated** using BSP (Binary Space Partitioning).
2. The agent starts at **Start** and must reach **Goal** on the target floor.
3. Along the way it may encounter **traps**, **doors/keys**, **health/ammo packs**, **stairs**, and **enemies** (patrol, chaser, sniper).
4. The agent observes the world through a **multimodal observation** (CNN visual + vector state) and picks one of **10 discrete actions**.
5. The environment follows the **Gymnasium API** (`reset` / `step`) and works with **Stable-Baselines3**.

The RL pipeline uses **curriculum learning**: the agent is trained sequentially through 9 levels from easy (`Crawl`) to very hard (`Master`), with a **mastery** threshold (win rate) before advancing to the next level.

---

## Key Features

| Area | Description |
|------|-------------|
| **Multi-floor** | Multiple floors linked by stairs; start and goal can be on different floors |
| **Procedural maps** | BSP + corridor carving + zone assignment; validates reachability and key/door logic |
| **Keys & doors** | Key on floor `k`, door blocking path on floor `d`; must collect key before opening door |
| **Traps & items** | Traps deal damage; health packs restore HP; ammo packs refill ammo |
| **3 enemy types** | Patrol (patrol + noise investigation), Chaser (A* pursuit), Sniper (laser + projectile) |
| **Combat & stealth** | Shooting (optional ricochet), dash consumes stamina, noise on shoot/dash |
| **Reward shaping** | Step penalty, explore/approach rewards, event-based bonuses and penalties |
| **RL training** | PPO / A2C, small CNN (~50K params), VecNormalize, mastery-gated curriculum |
| **Experiments** | 5 algorithm/hyperparameter variants + random baseline |
| **Utilities** | Map PNG rendering, demo GIF generation, Streamlit UI, Zeppelin notebooks |

---

## Tech Stack

- **Python 3.10+** (recommended)
- **[Gymnasium](https://gymnasium.farama.org/)** — RL environment API
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — PPO, A2C
- **[PyTorch](https://pytorch.org/)** — network backbone (CNN + MLP)
- **[NumPy](https://numpy.org/)** — grid and observations
- **[Matplotlib](https://matplotlib.org/)** — curriculum map rendering
- **[Streamlit](https://streamlit.io/)** — custom map playtest UI
- **[Pillow](https://python-pillow.org/)** — GIF frame rendering (used in `generate_gifs.py`)

---

## Project Structure

```
Agent-RF-Multifloor-Maze-Game/
├── README.md                          # This file
├── assets/demos/                      # Demo GIFs (optional, for README / reports)
└── multi_floor_maze/
    ├── mfm/
    │   └── env.py                     # Core: map generator, entities, MazeEnv, CURRICULUM
    ├── train.py                       # Single PPO training (baseline)
    ├── train_a2c.py                   # Single A2C training
    ├── train_experiments.py           # Train 5 experiments (exp1..exp5)
    ├── evaluate_experiments.py        # Eval 100 maps/level → JSON
    ├── generate_gifs.py               # Generate agent play GIFs
    ├── render_maps.py                 # Export curriculum map images (9 levels)
    ├── play_custom_map.py             # CLI: custom map + model → GIF
    ├── app_custom_map.py              # Streamlit UI
    ├── RL_ALP_GMM_Curriculum.ipynb    # Analysis / curriculum notebook (Zeppelin)
    ├── RL_Ver2_ZEPLEIN_UI.zpln        # Zeppelin notebook (UI + experiments)
    ├── ZEPPELIN_STRUCTURE.md          # Notebook evaluation structure guide
    ├── requirements.txt
    └── outputs/                       # Models, eval JSON, GIFs (gitignored — created when training)
        ├── exp1_ppo_final.zip
        ├── evals/exp1_ppo.json
        └── demos/
```

---

## Game Environment (`MazeEnv`)

Core file: `multi_floor_maze/mfm/env.py`.

### Tiles

| Tile | Meaning |
|------|---------|
| `FLOOR` | Walkable floor |
| `WALL` | Wall |
| `STAIR_UP` / `STAIR_DOWN` | Stairs up / down between floors |
| `GOAL` | Goal — reach to win |
| `START` | Spawn position |
| `TRAP` | Trap — lose HP when stepped on |
| `DOOR` | Door — requires a key to open |
| `KEY` | Key — increments key count |
| `HEALTH_PACK` | Restores 1 HP |
| `AMMO_PACK` | +2 ammo |

### Actions (10 total)

| ID | Action |
|----|--------|
| 0–3 | Move up / down / left / right |
| 4–7 | Shoot in 4 directions (costs 1 ammo, creates noise) |
| 8 | Dash — move up to 2 tiles in last direction (costs stamina, creates noise) |
| 9 | Use stairs (when standing on a stair tile) |

### Observation

Observation is a `Dict` with two parts:

**Visual** — tensor `(16, 9, 9)` with `view_radius=4` (9×9 window around the agent):

- 16 channels: wall, trap, door, stair up/down, goal, key, health, ammo, agent, 3 enemy types, bullet, laser, visited.

**Vector** — `(15,)` normalized to `[-1, 1]`:

- HP, ammo, stamina, keys, noise, floor ratio, direction to sub-goal/goal, target floor direction, last direction, time remaining, explored density, on stairs, nearest enemy proximity.

### Rewards (defaults)

| Event | Reward |
|-------|--------|
| Each step | -0.02 |
| Wall / invalid action | -0.12 |
| Win (goal) | +25.0 |
| Pick up key | +2.5 |
| Open door | +3.5 |
| Locked door (no key) | -0.15 |
| Kill enemy | +1.25 |
| Take damage | -2.0 |
| Death | -18.0 |
| First visit to new floor (stairs) | +0.75 |
| Explore new tile | +0.05 |
| Trap | -2.5 |
| Health / ammo pack | +0.75 / +0.50 |
| Approach sub-goal | +0.10 × Δdistance |
| Shoot | -0.04 |

Rewards can be overridden via `config["rewards"]`.

### Performance Optimizations

- **No BFS every step** for approach reward: distance to sub-goal is cached and recomputed only when key/door/floor state changes.
- Map generator caps choke-point candidates when placing doors (`_MAX_CHOKE_CANDIDATES = 14`).

### Enemies

| Type | Behavior | HP |
|------|----------|-----|
| **Patrol** | Patrols waypoints; investigates on sight or noise; melee when investigating | 1 |
| **Chaser** | Detects within radius / hears noise → pursues with A* | 2 |
| **Sniper** | Aligns with agent; charges laser for 3 steps then fires straight-line projectile | 1 |

The **ballistics** system handles agent and sniper projectiles (ricochet via `config["ricochet"]`). **Stealth** tracks noise from shooting/dashing so enemies react.

---

## Curriculum — 9 Difficulty Levels

Defined in `CURRICULUM` (`env.py`). The agent trains sequentially from level 1 → 9.

| Level | Name | Map | Floors | Highlights |
|-------|------|-----|--------|------------|
| 1 | **Crawl** | 7×7 | 1 | No enemies, no traps — basic movement |
| 2 | **Walk** | 9×9 | 1 | Larger map, still safe |
| 3 | **Trapper** | 9×9 | 1 | Traps introduced |
| 4 | **Locksmith** | 11×11 | 1 | 1 key/door pair |
| 5 | **Climber** | 11×11 | 2 | Multi-floor; start floor 1 → goal floor 0 |
| 6 | **Scout** | 11×11 | 2 | Ammo + 1 patrol enemy |
| 7 | **Operative** | 11×11 | 2 | Chaser + patrol + locks |
| 8 | **Elite** | 13×13 | 2 | Sniper + chaser + patrol + 2 locks |
| 9 | **Master** | 13×13 | 3 | 4 enemies, 3 locks, full traps/items — hardest level |

**Master (L9)** — typical configuration:

- 3 floors, start floor 2 → goal floor 0
- 4 enemies: sniper (F0), chaser (F1), patrol + chaser (F2)
- 3 locks: key/door on floors 0, 1, and 2
- HP 10, ammo 8, stamina 6, max 500 steps

---

## Procedural Map Generation

Class `ProceduralMapGenerator` in `env.py`:

1. **BSP split** — partition space into rooms.
2. **Carve rooms & corridors** — connect rooms with corridors (optional loops when no doors).
3. **Place stairs** — between consecutive floors.
4. **Start / Goal** — start on `start_floor`, goal on `goal_floor` (as far from start as possible).
5. **Locks** — door at BFS choke point; key placed in area reachable while door is closed.
6. **Traps / packs** — distributed by zone: spawn, passage, combat, treasure, exit.
7. **Validate** — floor ratio ≥ 20%, goal reachable, path long enough, correct key/door logic.

Each `reset(seed=...)` generates a new map (unless `options={"map_data": ...}` is passed).

---

## RL Training

### Policy Architecture

- **SmallDictExtractor**: 2-layer CNN (32→64) on visual + vector MLP (64→64) → 128-dim fusion.
- Policy head: `[128, 64]` for actor and critic.
- Total ~**50K parameters** (lightweight, trainable on CPU).

### Curriculum Plan (Mastery-Gated)

Each level has `timesteps` and a `mastery` threshold (eval win rate). If mastery is not met after the main step budget, training continues for up to 200K extra steps.

| Level | Timesteps | Mastery |
|-------|-----------|---------|
| 1 | 100K | 70% |
| 2 | 150K | 60% |
| 3 | 200K | 50% |
| 4 | 300K | 40% |
| 5 | 300K | 35% |
| 6 | 400K | 30% |
| 7 | 500K | 25% |
| 8 | 600K | 20% |
| 9 | 800K | 15% |

**Maximum ~3.35M timesteps** per run (excluding mastery retries).

### Training Scripts

| Script | Purpose | Default Device |
|--------|---------|----------------|
| `train.py` | PPO baseline, saves `ppo_level_*.zip`, `ppo_final.zip` | CPU |
| `train_a2c.py` | A2C baseline | CPU |
| `train_experiments.py` | 5 experiments, saves `exp*_final.zip` | CUDA |

All three use `DummyVecEnv` (4 parallel envs) and `VecNormalize` (reward normalization).

---

## 5 Comparison Experiments

Defined in `train_experiments.py`:

| ID | Name | Difference from baseline PPO |
|----|------|------------------------------|
| `exp1_ppo` | Default PPO | γ=0.99, ent_coef=0.01, net 128→64 |
| `exp2_a2c` | A2C | Same curriculum & extractor, A2C algorithm |
| `exp3_ppo_gamma095` | PPO γ=0.95 | Shorter discount horizon |
| `exp4_ppo_smallnet` | Small PPO net | features_dim=64, net [64,32] |
| `exp5_ppo_ent02` | PPO ent_coef=0.02 | More exploration |

For Zeppelin notebook details and evaluation charts, see `ZEPPELIN_STRUCTURE.md`.

---

## Evaluation & Results Export

```bash
cd multi_floor_maze
python evaluate_experiments.py
```

- Evaluates **100 random maps per level** (seed base 20000) for exp1..exp5 and a **random** baseline.
- Saves JSON to `outputs/evals/<exp_id>.json`.

Each JSON file includes: `mean_reward`, `std_reward`, `success_rate`, `returns[]`, `terminations` (win/timeout/death), `steps[]`, `regret`, `overall_mean_reward`, `overall_success_rate`.

Options:

```bash
python evaluate_experiments.py --only exp2_a2c
python evaluate_experiments.py --n-maps 50    # faster
```

---

## Demos, GIFs & Web UI

### Generate Agent Play GIFs

```bash
cd multi_floor_maze
python generate_gifs.py
python generate_gifs.py --exp exp2_a2c --level 9 --seed 42
```

GIFs are saved to `outputs/demos/` (grid + HUD style, inspired by Zeppelin UI).

### Render Curriculum Map Images

```bash
python render_maps.py
```

Images are saved to `outputs/level_maps/`.

### Custom Map CLI

```bash
python play_custom_map.py --width 13 --height 13 --n-floors 3 --traps 3 --locks 2 \
  --enemies "sniper:0,chaser:1,patrol:2" --model exp2_a2c --seed 99
```

### Streamlit UI

```bash
streamlit run app_custom_map.py
```

Open `http://localhost:8501` — customize the map, pick a model, watch the agent play, and download GIFs.

---

## Installation

```bash
git clone <repo-url>
cd Agent-RF-Multifloor-Maze-Game/multi_floor_maze

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Note:** `generate_gifs.py` requires **Pillow** (`pip install pillow`) — not listed in `requirements.txt`.

For GPU training (`train_experiments.py`), install a CUDA-enabled PyTorch build.

---

## Quick Start

### 1. Train a single PPO model

```bash
cd multi_floor_maze
python train.py
```

### 2. Train all experiments (long — run overnight)

```bash
python train_experiments.py
python train_experiments.py --only exp2_a2c
```

### 3. Evaluate & compare

```bash
python evaluate_experiments.py
```

### 4. Watch demos

```bash
python generate_gifs.py --exp exp2_a2c --level 9
# or
streamlit run app_custom_map.py
```

### Suggested workflow for reports / Zeppelin

1. `train_experiments.py` → produce 5 `.zip` models
2. `evaluate_experiments.py` → produce eval JSON
3. `render_maps.py` → map images for all 9 levels
4. `generate_gifs.py` → demo GIFs (copy to `assets/demos/` to display in README)
5. Open `RL_Ver2_ZEPLEIN_UI.zpln` / `RL_ALP_GMM_Curriculum.ipynb` — plot charts per `ZEPPELIN_STRUCTURE.md`

---

## Outputs

Directory `multi_floor_maze/outputs/` (gitignored):

| Path | Contents |
|------|----------|
| `ppo_final.zip`, `ppo_level_*.zip` | Checkpoints from `train.py` |
| `exp1_ppo_final.zip` … `exp5_ppo_ent02_final.zip` | Experiment models |
| `exp*_level_*.zip` | Per-level checkpoints |
| `evals/*.json` | Evaluation results |
| `demos/*.gif` | Agent play GIFs |
| `level_maps/*.png` | Curriculum map images |

---

## Hardest Level Demo

Agent (A2C, exp2) playing **Master** (13×13, 3 floors, 4 enemies, 3 locks):

| Seed 42 | Seed 142 | Seed 242 |
|--------|---------|----------|
| ![exp2_a2c L9 seed42](assets/demos/exp2_a2c_L9_Master_seed42_WIN.gif) | ![exp2_a2c L9 seed142](assets/demos/exp2_a2c_L9_Master_seed142_WIN.gif) | ![exp2_a2c L9 seed242](assets/demos/exp2_a2c_L9_Master_seed242_WIN.gif) |

> Demo GIFs live in `assets/demos/` (or generate with `generate_gifs.py` and copy there). If missing, run:
> ```bash
> python generate_gifs.py --exp exp2_a2c --level 9 --seed 42
> ```

---

## License

MIT
>>>>>>> theirs
