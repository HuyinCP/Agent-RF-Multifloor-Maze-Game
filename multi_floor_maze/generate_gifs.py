"""
Generate GIF demos of trained agents (PPO / DQN / A2C) playing each level.
Style: Zeppelin-UI inspired — grid map + side HUD panel with live stats.

Usage:
  # Chế độ mặc định: quét outputs/ tìm ppo_level_*.zip, dqn_level_*.zip và
  # các thư mục exp* (exp1_ppo, exp2_a2c, ...), với mỗi checkpoint chạy 3 seed → lưu GIF vào outputs/demos/
  python generate_gifs.py

  # Chỉ một exp + level (vd: exp2_a2c, level 5), 1 seed
  python generate_gifs.py --exp exp2_a2c --level 5

  # Chỉ một exp, tất cả level 1–9, 3 seed (giống mặc định nhưng chỉ exp đó)
  python generate_gifs.py --exp exp2_a2c

  # Chỉ định file model trực tiếp
  python generate_gifs.py --model outputs/exp2_a2c/exp2_a2c_final.zip --level 3 --seed 42

  # Ghi đè kích thước map lúc test (vd: map 15x15, 2 tầng)
  python generate_gifs.py --exp exp2_a2c --level 9 --width 15 --height 15 --n-floors 2
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mfm.env import (
    CURRICULUM,
    MazeEnv,
    SniperEnemy,
    Tile,
)

# ── Config ───────────────────────────────────────────────────
CELL = 48
HUD_W = 220
FONT_SIZE_TILE = int(CELL * 0.65)
FONT_SIZE_UNIT = int(CELL * 0.48)
FONT_SIZE_HUD = 16
FONT_SIZE_HUD_TITLE = 20
GIF_DURATION = 350  # ms mỗi frame (tăng → GIF chạy chậm hơn; 200 nhanh, 400–500 chậm rõ)
SEED = 42
OUTPUT_DIR = ROOT / "outputs" / "demos"

TILE_COLOR = {
    Tile.FLOOR:       (250, 250, 250),
    Tile.WALL:        (40, 40, 40),
    Tile.STAIR_UP:    (180, 180, 255),
    Tile.STAIR_DOWN:  (150, 150, 220),
    Tile.GOAL:        (255, 255, 100),
    Tile.TRAP:        (255, 80, 80),
    Tile.START:       (180, 255, 180),
    Tile.DOOR:        (180, 140, 100),
    Tile.KEY:         (220, 180, 80),
    Tile.HEALTH_PACK: (255, 200, 200),
    Tile.AMMO_PACK:   (240, 240, 150),
}

TILE_LABEL = {
    Tile.STAIR_UP: "SU", Tile.STAIR_DOWN: "SD",
    Tile.GOAL: "G", Tile.TRAP: "X", Tile.START: "S",
    Tile.DOOR: "D", Tile.KEY: "K",
    Tile.HEALTH_PACK: "+", Tile.AMMO_PACK: "A",
}

ENEMY_STYLE = {
    "patrol": ((200, 100, 200), "P"),
    "chaser": ((255, 50, 50), "E"),
    "sniper": ((255, 165, 0), "S"),
}


def _get_font(size):
    paths = [
        "Arial.ttf", "arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TILE = _get_font(FONT_SIZE_TILE)
FONT_UNIT = _get_font(FONT_SIZE_UNIT)
FONT_HUD = _get_font(FONT_SIZE_HUD)
FONT_HUD_TITLE = _get_font(FONT_SIZE_HUD_TITLE)
FONT_HUD_BIG = _get_font(28)


def render_frame(env, step_num=0, total_reward=0.0, model_name="PPO"):
    """Render 1 frame: grid map + HUD panel on the right."""
    g = env.floors[env.agent.floor]
    h, w = g.shape
    map_w, map_h = w * CELL, h * CELL
    img_w = map_w + HUD_W
    img_h = max(map_h, 400)

    img = Image.new("RGB", (img_w, img_h), (245, 245, 245))
    d = ImageDraw.Draw(img)

    # ── Draw tiles ───────────────────────────────────────────
    for y in range(h):
        for x in range(w):
            t = int(g[y][x])
            x1, y1 = x * CELL, y * CELL
            c = TILE_COLOR.get(t, (250, 250, 250))
            d.rectangle([x1, y1, x1 + CELL - 2, y1 + CELL - 2],
                        fill=c, outline=(150, 150, 150))
            lb = TILE_LABEL.get(t)
            if lb:
                d.text((x1 + CELL // 2, y1 + CELL // 2), lb,
                       fill=(0, 0, 0), font=FONT_TILE, anchor="mm")

    # ── Sniper laser paths ───────────────────────────────────
    for e in env.enemies:
        if isinstance(e, SniperEnemy) and e.charging and e.floor == env.agent.floor:
            for lx, ly in e.laser_path:
                d.rectangle([lx * CELL + CELL // 3, ly * CELL + CELL // 3,
                             lx * CELL + 2 * CELL // 3, ly * CELL + 2 * CELL // 3],
                            fill=(255, 50, 50, 180))

    # ── Enemies ──────────────────────────────────────────────
    for e in env.enemies:
        if e.alive and e.floor == env.agent.floor:
            ec, lbl = ENEMY_STYLE.get(e.etype, ((150, 150, 150), "?"))
            d.ellipse([e.x * CELL + 4, e.y * CELL + 4,
                       e.x * CELL + CELL - 6, e.y * CELL + CELL - 6],
                      fill=ec, outline=(0, 0, 0), width=2)
            d.text((e.x * CELL + CELL // 2, e.y * CELL + CELL // 2), lbl,
                   fill=(255, 255, 255), font=FONT_UNIT, anchor="mm")

    # ── Projectiles ──────────────────────────────────────────
    for bx, by, owner in env.ballistics.positions(env.agent.floor):
        bc = (200, 200, 0) if owner == "agent" else (255, 0, 0)
        d.ellipse([bx * CELL + CELL // 3, by * CELL + CELL // 3,
                   bx * CELL + 2 * CELL // 3, by * CELL + 2 * CELL // 3],
                  fill=bc)

    # ── Agent ────────────────────────────────────────────────
    a = env.agent
    if a.hp <= 1:
        ac = (255, 100, 100)
    elif a.keys > 0:
        ac = (100, 200, 100)
    else:
        ac = (100, 150, 255)
    ax1, ay1 = a.x * CELL + 5, a.y * CELL + 5
    d.ellipse([ax1, ay1, ax1 + CELL - 10, ay1 + CELL - 10],
              fill=ac, outline=(0, 0, 0), width=2)
    d.text((a.x * CELL + CELL // 2, a.y * CELL + CELL // 2), "@",
           fill=(0, 0, 0), font=FONT_UNIT, anchor="mm")

    # ── HUD Panel ────────────────────────────────────────────
    hx = map_w + 8
    d.rectangle([map_w, 0, img_w, img_h], fill=(30, 30, 45))

    d.text((hx + HUD_W // 2 - 8, 12), model_name,
           fill=(255, 255, 255), font=FONT_HUD_TITLE, anchor="mt")

    level_cfg = env.cfg
    level_name = level_cfg.get("name", "?")
    d.text((hx + HUD_W // 2 - 8, 38), f"Level: {level_name}",
           fill=(180, 180, 200), font=FONT_HUD, anchor="mt")

    hy = 70
    stats = [
        ("Floor",   f"{a.floor + 1}/{env.nf}", (180, 180, 255)),
        ("Steps",   f"{a.steps}/{env.max_steps}", (200, 200, 200)),
        ("Score",   f"{total_reward:+.1f}", (100, 255, 100) if total_reward >= 0 else (255, 100, 100)),
        ("HP",      f"{a.hp}/{a.max_hp}", _hp_color(a.hp, a.max_hp)),
        ("Ammo",    f"{a.ammo}/{a.max_ammo}", (240, 240, 150)),
        ("Stamina", f"{a.stamina}/{a.max_stamina}", (200, 200, 200)),
        ("Keys",    str(a.keys), (220, 180, 80)),
        ("Noise",   str(a.noise_level), (255, 200, 150)),
    ]

    for label, val, color in stats:
        d.text((hx + 6, hy), label, fill=(140, 140, 160), font=FONT_HUD)
        d.text((hx + HUD_W - 16, hy), val, fill=color, font=FONT_HUD_BIG, anchor="rt")
        hy += 36

    # ── Legend ────────────────────────────────────────────────
    hy += 10
    d.text((hx + 6, hy), "── Legend ──", fill=(120, 120, 140), font=FONT_HUD)
    hy += 24
    legend = [
        ((100, 150, 255), "@  Agent"),
        ((255, 255, 100), "G  Goal"),
        ((255, 80, 80), "X  Trap"),
        ((180, 140, 100), "D  Door"),
        ((220, 180, 80), "K  Key"),
        ((180, 180, 255), "SU Stair Up"),
        ((200, 100, 200), "P  Patrol"),
        ((255, 50, 50), "E  Chaser"),
        ((255, 165, 0), "S  Sniper"),
    ]
    for color, text in legend:
        d.rectangle([hx + 6, hy + 1, hx + 18, hy + 13], fill=color, outline=(80, 80, 80))
        d.text((hx + 24, hy), text, fill=(180, 180, 190), font=FONT_HUD)
        hy += 18

    return img


def _hp_color(hp, max_hp):
    ratio = hp / max(max_hp, 1)
    if ratio > 0.6:
        return (100, 255, 100)
    if ratio > 0.3:
        return (255, 200, 80)
    return (255, 80, 80)


# ── Feature extractor (must match training) ──────────────────

class SmallDictExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space["visual"].shape[0]
        vec_dim = observation_space["vector"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.vec_net = nn.Sequential(
            nn.Linear(vec_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(128, features_dim), nn.ReLU())

    def forward(self, obs):
        vis = self.cnn(obs["visual"])
        vec = self.vec_net(obs["vector"])
        return self.fusion(torch.cat([vis, vec], dim=1))


POLICY_KWARGS = dict(
    features_extractor_class=SmallDictExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


class SmallDictExtractor64(BaseFeaturesExtractor):
    """Mạng nhỏ (features_dim=64) — dùng cho exp4_ppo_smallnet."""
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space["visual"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        vec_dim = observation_space["vector"].shape[0]
        self.vec_net = nn.Sequential(
            nn.Linear(vec_dim, 48), nn.ReLU(),
            nn.Linear(48, 48), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(48 + 48, features_dim), nn.ReLU())

    def forward(self, obs):
        vis = self.cnn(obs["visual"])
        vec = self.vec_net(obs["vector"])
        return self.fusion(torch.cat([vis, vec], dim=1))


CUSTOM = {
    "SmallDictExtractor": SmallDictExtractor,
    "SmallDictExtractor64": SmallDictExtractor64,
}


def load_model(model_type, path):
    """Load theo loại: PPO, A2C, hoặc DQN."""
    if model_type == "PPO":
        return PPO.load(str(path), device="cpu", custom_objects=CUSTOM)
    if model_type == "A2C":
        return A2C.load(str(path), device="cpu", custom_objects=CUSTOM)
    if model_type == "DQN":
        return DQN.load(str(path), device="cpu", custom_objects=CUSTOM)
    raise ValueError(f"Unknown model type: {model_type}")


def run_episode(model, level_cfg, model_name, seed=SEED):
    """Run 1 episode, collect frames + stats."""
    env = MazeEnv(config=copy.deepcopy(level_cfg))
    obs, _ = env.reset(seed=seed)

    frames = [render_frame(env, 0, 0.0, model_name)]
    total_r = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_r += reward
        done = term or trunc
        frames.append(render_frame(env, env.agent.steps, total_r, model_name))

    return frames, info, total_r


def save_gif(frames, path, duration=GIF_DURATION):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > 1:
        frames[0].save(
            str(path), format="GIF", save_all=True,
            append_images=frames[1:], duration=duration, loop=0,
        )
    else:
        frames[0].save(str(path))


DEMO_SEEDS = [42, 142, 242]


def _collect_jobs(output_path: Path):
    """Thu thập (model_type, level, ckpt_path). model_type dùng cho load_model (PPO/A2C/DQN)."""
    jobs = []
    # PPO / DQN đặt tên ppo_level_*.zip, dqn_level_*.zip ngay trong outputs/
    for lv in range(1, 10):
        ckpt = output_path / f"ppo_level_{lv}.zip"
        if ckpt.exists():
            jobs.append(("PPO", lv, ckpt))
    for lv in range(1, 5):
        ckpt = output_path / f"dqn_level_{lv}.zip"
        if ckpt.exists():
            jobs.append(("DQN", lv, ckpt))
    # Các thí nghiệm exp*: outputs/exp2_a2c/exp2_a2c_final.zip (chạy trên mọi level 1–9)
    for d in output_path.iterdir():
        if not d.is_dir() or not d.name.startswith("exp"):
            continue
        exp_id = d.name
        load_type = "A2C" if "a2c" in exp_id else "PPO"
        final_zip = d / f"{exp_id}_final.zip"
        if final_zip.exists():
            for lv in range(1, 10):
                jobs.append((load_type, lv, final_zip))
    # outputs/exp2_a2c_final.zip (file đặt ngay trong outputs/)
    for f in output_path.glob("exp*_final.zip"):
        if f in [j[2] for j in jobs]:
            continue
        exp_id = f.stem.replace("_final", "")
        load_type = "A2C" if "a2c" in exp_id else "PPO"
        for lv in range(1, 10):
            jobs.append((load_type, lv, f))
    return jobs


def _run_one(model_type, level, ckpt_path, seeds, output_dir, summary, duration=GIF_DURATION, frame_skip=1, size_override=None):
    """size_override: dict optional với keys width, height, n_floors — ghi đè config level khi test."""
    level_cfg = copy.deepcopy(CURRICULUM[level])
    if size_override:
        if "width" in size_override:
            level_cfg["width"] = size_override["width"]
        if "height" in size_override:
            level_cfg["height"] = size_override["height"]
        if "n_floors" in size_override:
            level_cfg["n_floors"] = size_override["n_floors"]
    level_name = level_cfg["name"]
    label = ckpt_path.stem.replace("_final", "").replace(f"_level_{level}", "")
    try:
        model = load_model(model_type, ckpt_path)
    except Exception as e:
        print(f"  {label} L{level} ({level_name}) — LOAD ERROR: {e}")
        return
    for seed in seeds:
        print(f"  {label} L{level} {level_name} seed={seed} ... ", end="", flush=True)
        try:
            frames, info, total_r = run_episode(model, level_cfg, label, seed=seed)
            if frame_skip > 1:
                frames = frames[::frame_skip]
            win = info.get("win", False)
            steps = info.get("steps", len(frames) - 1)
            tag = "WIN" if win else "LOSE"
            gif_name = f"{label}_L{level}_{level_name}_seed{seed}_{tag}.gif"
            save_gif(frames, output_dir / gif_name, duration=duration)
            summary.append({
                "model": label, "level": level, "name": level_name,
                "seed": seed, "win": win, "steps": steps,
                "reward": round(total_r, 2), "frames": len(frames),
                "gif": gif_name,
            })
            print(f"{tag} | steps={steps} | reward={total_r:+.2f} | {len(frames)} frames")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    ap = argparse.ArgumentParser(description="Tạo GIF demo agent chơi từng level (PPO/DQN/A2C).")
    ap.add_argument("--exp", type=str, default=None,
                    help="Chỉ chạy một exp (vd: exp2_a2c). Model: outputs/<exp>/<exp>_final.zip")
    ap.add_argument("--model", type=str, default=None,
                    help="Đường dẫn file .zip model (vd: outputs/exp2_a2c/exp2_a2c_final.zip)")
    ap.add_argument("--level", type=int, default=None, choices=list(range(1, 10)),
                    help="Chỉ chạy một level (1–9). Nếu không đặt thì chạy tất cả level.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Một seed (vd: 42). Nếu không đặt thì dùng 3 seed: 42, 142, 242.")
    ap.add_argument("--output", type=str, default=None,
                    help="Thư mục lưu GIF (mặc định: outputs/demos)")
    ap.add_argument("--duration", type=int, default=GIF_DURATION,
                    help=f"Ms mỗi frame (tăng → GIF chạy chậm hơn). Mặc định: {GIF_DURATION}")
    ap.add_argument("--frame-skip", type=int, default=1, metavar="N",
                    help="Chỉ lưu mỗi N frame (1=all, 2=half…). Mặc định: 1")
    ap.add_argument("--width", type=int, default=None, metavar="W",
                    help="Ghi đè chiều rộng map (số ô) khi test. Không đặt = dùng theo level.")
    ap.add_argument("--height", type=int, default=None, metavar="H",
                    help="Ghi đè chiều cao map (số ô) khi test. Không đặt = dùng theo level.")
    ap.add_argument("--n-floors", type=int, default=None, metavar="F",
                    help="Ghi đè số tầng map khi test. Không đặt = dùng theo level.")
    args = ap.parse_args()

    if args.frame_skip < 1:
        args.frame_skip = 1

    size_override = {}
    if args.width is not None:
        size_override["width"] = args.width
    if args.height is not None:
        size_override["height"] = args.height
    if args.n_floors is not None:
        size_override["n_floors"] = args.n_floors
    size_override = size_override if size_override else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ROOT / "outputs"
    out_dir = Path(args.output).resolve() if args.output else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else DEMO_SEEDS
    duration = args.duration
    frame_skip = args.frame_skip

    # Chế độ một lần chạy: --exp hoặc --model
    if args.exp or args.model:
        if args.model:
            p = Path(args.model).resolve()
            if not p.is_absolute():
                p = (ROOT / args.model).resolve()
            if not p.exists():
                print(f"Không tìm thấy: {p}")
                return
            model_type = "A2C" if "a2c" in p.name.lower() else "PPO"
            label = p.stem.replace("_final", "")
            levels = [args.level] if args.level is not None else list(range(1, 10))
            jobs = [(model_type, lv, p) for lv in levels]
        else:
            # --exp exp2_a2c
            exp_id = args.exp
            final_zip = output_path / exp_id / f"{exp_id}_final.zip"
            if not final_zip.exists():
                final_zip = output_path / f"{exp_id}_final.zip"
            if not final_zip.exists():
                print(f"Không tìm thấy model: {final_zip}")
                return
            model_type = "A2C" if "a2c" in exp_id else "PPO"
            levels = [args.level] if args.level is not None else list(range(1, 10))
            jobs = [(model_type, lv, final_zip) for lv in levels]
        summary = []
        for mt, lv, ckpt in jobs:
            _run_one(mt, lv, ckpt, seeds, out_dir, summary, duration=duration, frame_skip=frame_skip, size_override=size_override)
        if summary:
            print(f"\nGIFs saved to: {out_dir}")
        return

    # Chế độ mặc định: quét tất cả
    jobs = _collect_jobs(output_path)
    total_gifs = len(jobs) * len(seeds)
    print(f"Generating {total_gifs} GIF demos ({len(jobs)} checkpoints × {len(seeds)} seeds)...\n")
    summary = []
    for model_type, level, ckpt_path in jobs:
        _run_one(model_type, level, ckpt_path, seeds, out_dir, summary, duration=duration, frame_skip=frame_skip, size_override=size_override)

    print(f"\n{'=' * 75}")
    print(f"{'Model':<12} {'Lv':>2} {'Name':<12} {'Seed':>5} {'Result':>6} {'Steps':>6} {'Reward':>8}")
    print(f"{'-' * 75}")
    current_key = None
    for s in summary:
        key = (s["model"], s["level"])
        if key != current_key:
            if current_key is not None:
                wins = [x for x in summary if (x["model"], x["level"]) == current_key and x.get("win")]
                total = [x for x in summary if (x["model"], x["level"]) == current_key]
                print(f"{'':>32} win rate: {len(wins)}/{len(total)}")
            current_key = key
        tag = "WIN" if s["win"] else "LOSE"
        print(f"{s['model']:<12} {s['level']:>2} {s['name']:<12} {s['seed']:>5} {tag:>6} {s['steps']:>6} {s['reward']:>+8.2f}")
    if current_key:
        wins = [x for x in summary if (x["model"], x["level"]) == current_key and x.get("win")]
        total = [x for x in summary if (x["model"], x["level"]) == current_key]
        print(f"{'':>32} win rate: {len(wins)}/{len(total)}")
    print(f"{'=' * 75}")
    print(f"\nGIFs saved to: {out_dir}")


if __name__ == "__main__":
    main()
