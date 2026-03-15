"""
Train nhiều thí nghiệm (PPO, A2C, PPO biến thể) — cùng curriculum.
Dùng để treo đêm: chạy lần lượt 5 mô hình, lưu exp1_ppo_final.zip ... exp5_ppo_ent02_final.zip.

Usage:
  python train_experiments.py              # train tất cả
  python train_experiments.py --only exp1  # chỉ exp1_ppo
  python train_experiments.py --only exp2  # chỉ exp2_a2c
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from mfm.env import MazeEnv, CURRICULUM

DEVICE = "cuda"
N_ENVS = 4
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CURRICULUM_PLAN = [
    {"level": 1, "timesteps": 100_000, "mastery": 0.70},
    {"level": 2, "timesteps": 150_000, "mastery": 0.60},
    {"level": 3, "timesteps": 200_000, "mastery": 0.50},
    {"level": 4, "timesteps": 300_000, "mastery": 0.40},
    {"level": 5, "timesteps": 300_000, "mastery": 0.35},
    {"level": 6, "timesteps": 400_000, "mastery": 0.30},
    {"level": 7, "timesteps": 500_000, "mastery": 0.25},
    {"level": 8, "timesteps": 600_000, "mastery": 0.20},
    {"level": 9, "timesteps": 800_000, "mastery": 0.15},
]


# ═══════════════════════════════════════════════════════════════
# Feature extractors
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
    """Mạng nhỏ hơn (features_dim=64, net nhỏ) — thí nghiệm 4."""
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


# ═══════════════════════════════════════════════════════════════
# Env & callback
# ═══════════════════════════════════════════════════════════════

def make_env(level_config, seed=None):
    def _init():
        env = MazeEnv(config=copy.deepcopy(level_config))
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def make_vec_env(level_config, n_envs=N_ENVS, normalize=True):
    envs = DummyVecEnv([make_env(level_config, seed=i * 1000) for i in range(n_envs)])
    if normalize:
        envs = VecNormalize(envs, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    return envs


class LogCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.ep_rewards = []
        self.ep_wins = []
        self._running_rewards = None

    def _on_training_start(self):
        self._running_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        self._running_rewards += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.ep_rewards.append(float(self._running_rewards[i]))
                self._running_rewards[i] = 0.0
                infos = self.locals.get("infos", [])
                if i < len(infos) and "win" in infos[i]:
                    self.ep_wins.append(1.0 if infos[i]["win"] else 0.0)
        if self.n_calls % self.log_interval == 0 and self.ep_rewards:
            recent_r = self.ep_rewards[-100:]
            recent_w = self.ep_wins[-100:]
            print(f"  step={self.num_timesteps:>8d} | mean_r={np.mean(recent_r):>7.2f} | win={np.mean(recent_w) if recent_w else 0:.2%}")
        return True


def evaluate(model, level_config, n_episodes=30):
    rewards, wins = [], []
    for seed in range(9000, 9000 + n_episodes):
        env = MazeEnv(config=copy.deepcopy(level_config))
        obs, _ = env.reset(seed=seed)
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        wins.append(1.0 if info.get("win", False) else 0.0)
    return float(np.mean(rewards)), float(np.mean(wins))


# ═══════════════════════════════════════════════════════════════
# Định nghĩa 5 thí nghiệm
# ═══════════════════════════════════════════════════════════════

def build_ppo(env, *, gamma=0.99, ent_coef=0.01, features_dim=128, net_arch=None):
    if net_arch is None:
        net_arch = dict(pi=[128, 64], vf=[128, 64])
    extractor = SmallDictExtractor if features_dim == 128 else SmallDictExtractor64
    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=net_arch,
        activation_fn=nn.ReLU,
    )
    return PPO(
        "MultiInputPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        verbose=0,
    )


def build_a2c(env):
    policy_kwargs = dict(
        features_extractor_class=SmallDictExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
    )
    return A2C(
        "MultiInputPolicy", env,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        verbose=0,
    )


EXPERIMENTS = {
    "exp1_ppo": {
        "name": "Thí nghiệm 1: PPO (mặc định)",
        "build": lambda env: build_ppo(env),
    },
    "exp2_a2c": {
        "name": "Thí nghiệm 2: A2C",
        "build": build_a2c,
    },
    "exp3_ppo_gamma095": {
        "name": "Thí nghiệm 3: PPO gamma=0.95 (discount ngắn hơn)",
        "build": lambda env: build_ppo(env, gamma=0.95),
    },
    "exp4_ppo_smallnet": {
        "name": "Thí nghiệm 4: PPO mạng nhỏ (features_dim=64)",
        "build": lambda env: build_ppo(env, features_dim=64, net_arch=dict(pi=[64, 32], vf=[64, 32])),
    },
    "exp5_ppo_ent02": {
        "name": "Thí nghiệm 5: PPO ent_coef=0.02 (khám phá nhiều hơn)",
        "build": lambda env: build_ppo(env, ent_coef=0.02),
    },
}


def train_one(exp_id):
    spec = EXPERIMENTS[exp_id]
    print("\n" + "=" * 60)
    print(spec["name"])
    print("=" * 60)

    first_cfg = copy.deepcopy(CURRICULUM[CURRICULUM_PLAN[0]["level"]])
    vec_env = make_vec_env(first_cfg)
    model = spec["build"](vec_env)

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Params: {n_params:,}")

    for stage_idx, stage in enumerate(CURRICULUM_PLAN):
        level = stage["level"]
        timesteps = stage["timesteps"]
        mastery = stage["mastery"]
        level_cfg = copy.deepcopy(CURRICULUM[level])
        level_name = level_cfg.get("name", f"L{level}")

        print(f"\n--- Level {level} {level_name} | {timesteps:,} steps | mastery {mastery:.0%} ---")
        vec_env = make_vec_env(level_cfg)
        model.set_env(vec_env)

        t0 = time.time()
        model.learn(total_timesteps=timesteps, callback=LogCallback(log_interval=10000), reset_num_timesteps=False)
        elapsed = time.time() - t0

        mean_r, win_rate = evaluate(model, level_cfg, n_episodes=30)
        print(f"  Done {elapsed:.0f}s | reward={mean_r:.2f} | win={win_rate:.1%}")

        ckpt = os.path.join(OUTPUT_DIR, f"{exp_id}_level_{level}.zip")
        model.save(ckpt)

        if win_rate < mastery:
            extra = min(timesteps, 200_000)
            print(f"  Below mastery, +{extra:,} steps")
            model.learn(total_timesteps=extra, callback=LogCallback(log_interval=10000), reset_num_timesteps=False)
            mean_r2, wr2 = evaluate(model, level_cfg, n_episodes=30)
            print(f"  Retry: reward={mean_r2:.2f} | win={wr2:.1%}")
            model.save(ckpt)

    final_path = os.path.join(OUTPUT_DIR, f"{exp_id}_final.zip")
    model.save(final_path)
    print(f"\nSaved: {final_path}")

    print("\nFinal eval:")
    for stage in CURRICULUM_PLAN:
        level = stage["level"]
        cfg = copy.deepcopy(CURRICULUM[level])
        mr, wr = evaluate(model, cfg, n_episodes=30)
        print(f"  L{level:>2d} {cfg['name']:<12s} | reward={mr:>7.2f} | win={wr:.1%}")
    return final_path


def main():
    ap = argparse.ArgumentParser(description="Train nhiều thí nghiệm PPO/A2C")
    ap.add_argument("--only", type=str, default=None, choices=list(EXPERIMENTS),
                    help="Chỉ chạy 1 thí nghiệm (exp1_ppo, exp2_a2c, ...)")
    args = ap.parse_args()

    to_run = [args.only] if args.only else list(EXPERIMENTS)
    for exp_id in to_run:
        train_one(exp_id)
    print("\n" + "=" * 60)
    print("TẤT CẢ THÍ NGHIỆM HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    main()
