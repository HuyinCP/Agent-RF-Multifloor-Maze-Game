"""
Đánh giá tất cả thí nghiệm (exp1..exp5 + random) trên 100 map/level.
Lưu JSON đầy đủ: expected return, returns[], terminations (win/timeout/death), steps — để vẽ biểu đồ trong Zeppelin.

Usage:
  python evaluate_experiments.py                    # eval tất cả
  python evaluate_experiments.py --only exp1_ppo   # chỉ exp1
  python evaluate_experiments.py --n-maps 50       # nhanh hơn
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mfm.env import CURRICULUM, MazeEnv

OUTPUT_DIR = ROOT / "outputs"
EVALS_DIR = OUTPUT_DIR / "evals"
DEVICE = "cpu"
N_MAPS = 100
SEED_BASE = 20000


# ═══════════════════════════════════════════════════════════════
# Feature extractors (khớp train_experiments.py)
# ═══════════════════════════════════════════════════════════════

class SmallDictExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space["visual"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        vec_dim = observation_space["vector"].shape[0]
        self.vec_net = nn.Sequential(
            nn.Linear(vec_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(64 + 64, features_dim), nn.ReLU())

    def forward(self, obs):
        vis = self.cnn(obs["visual"])
        vec = self.vec_net(obs["vector"])
        return self.fusion(torch.cat([vis, vec], dim=1))


class SmallDictExtractor64(BaseFeaturesExtractor):
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


def load_model(exp_id, path_override=None):
    if path_override:
        p = Path(path_override)
    else:
        p = OUTPUT_DIR / f"{exp_id}_final.zip"
    if not p.exists():
        return None
    try:
        if "a2c" in exp_id:
            return A2C.load(str(p), device=DEVICE, custom_objects=CUSTOM)
        return PPO.load(str(p), device=DEVICE, custom_objects=CUSTOM)
    except Exception:
        return PPO.load(str(p), device=DEVICE, custom_objects=CUSTOM)


def evaluate_one_level(model_or_none, level, n_maps, seed_base, label="model"):
    cfg = copy.deepcopy(CURRICULUM[level])
    env = MazeEnv(config=cfg)
    seeds = [seed_base + i for i in range(n_maps)]
    returns_list = []
    terminations = {"win": 0, "timeout": 0, "death": 0}
    steps_list = []
    max_theoretical_list = []

    for seed in seeds:
        env = MazeEnv(config=cfg)
        obs, _ = env.reset(seed=seed)
        total_r = 0.0
        steps = 0
        done = False
        while not done:
            if model_or_none is None:
                action = env.action_space.sample()
            else:
                action, _ = model_or_none.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total_r += r
            steps += 1
            done = term or trunc
        returns_list.append(total_r)
        steps_list.append(steps)
        if info.get("win", False):
            terminations["win"] += 1
        elif trunc:
            terminations["timeout"] += 1
        else:
            terminations["death"] += 1
        mt = info.get("max_theoretical_reward")
        if mt is not None:
            max_theoretical_list.append(mt)

    mean_reward = float(np.mean(returns_list))
    std_reward = float(np.std(returns_list))
    success_rate = terminations["win"] / n_maps
    mean_steps = float(np.mean(steps_list))
    # Regret: nếu có max_theoretical thì regret = mean(max_theoretical - return), không thì dùng 1 - success_rate
    if max_theoretical_list:
        mean_max = float(np.mean(max_theoretical_list))
        regret = mean_max - mean_reward
    else:
        regret = 1.0 - success_rate  # proxy

    return {
        "level": level,
        "name": cfg.get("name", f"L{level}"),
        "n_maps": n_maps,
        "seed_base": seed_base,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "returns": returns_list,
        "terminations": terminations,
        "steps": steps_list,
        "regret": regret,
        "mean_max_theoretical": float(np.mean(max_theoretical_list)) if max_theoretical_list else None,
    }


def evaluate_experiment(exp_id, n_maps, seed_base):
    if exp_id == "random":
        model = None
        label = "Random"
    else:
        model = load_model(exp_id)
        if model is None:
            print(f"  Skip {exp_id}: no model file")
            return None
        label = exp_id
    levels = list(range(1, 10))
    results = []
    for level in levels:
        row = evaluate_one_level(model, level, n_maps, seed_base, label=label)
        results.append(row)
        print(f"  L{level} {row['name']}: reward={row['mean_reward']:+.2f} ± {row['std_reward']:.2f} | "
              f"win={row['success_rate']:.1%} | term win/timeout/death = {row['terminations']['win']}/{row['terminations']['timeout']}/{row['terminations']['death']}")
    overall_win = np.mean([r["success_rate"] for r in results])
    overall_rew = np.mean([r["mean_reward"] for r in results])
    return {
        "experiment_id": exp_id,
        "label": label,
        "n_maps_per_level": n_maps,
        "seed_base": seed_base,
        "results": results,
        "overall_mean_reward": float(overall_rew),
        "overall_success_rate": float(overall_win),
    }


def main():
    ap = argparse.ArgumentParser(description="Đánh giá tất cả thí nghiệm (exp1..exp5 + random)")
    ap.add_argument("--only", type=str, default=None,
                    help="Chỉ eval 1 (exp1_ppo, exp2_a2c, ..., random)")
    ap.add_argument("--n-maps", type=int, default=N_MAPS, help="Số map mỗi level")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE, help="Seed base")
    args = ap.parse_args()

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    experiment_ids = ["exp1_ppo", "exp2_a2c", "exp3_ppo_gamma095", "exp4_ppo_smallnet", "exp5_ppo_ent02", "random"]
    if args.only:
        experiment_ids = [args.only]

    for exp_id in experiment_ids:
        print("\n" + "=" * 60)
        print(f"Eval: {exp_id}")
        print("=" * 60)
        data = evaluate_experiment(exp_id, args.n_maps, args.seed_base)
        if data is None:
            continue
        out_path = EVALS_DIR / f"{exp_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Lưu: {out_path}")
        print(f"  Overall: mean_reward={data['overall_mean_reward']:+.2f} | win_rate={data['overall_success_rate']:.1%}")

    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()
