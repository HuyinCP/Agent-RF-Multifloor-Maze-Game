"""
Multi-Floor Maze RL Training — Mac ARM M2 Optimized
PPO with small CNN, mastery-gated curriculum, VecNormalize.

Usage: python train.py
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

from mfm.env import MazeEnv, CURRICULUM

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DEVICE = "cpu"
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
# Feature Extractor — Small & Efficient
# ═══════════════════════════════════════════════════════════════

class SmallDictExtractor(BaseFeaturesExtractor):
    """CNN(32->64) + AdaptivePool + Vector MLP -> 128 dim fusion.
    Total params: ~50K instead of ~10M.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        vis_shape = observation_space["visual"].shape
        n_ch = vis_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        cnn_out = 64

        vec_dim = observation_space["vector"].shape[0]
        self.vec_net = nn.Sequential(
            nn.Linear(vec_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        vec_out = 64

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + vec_out, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        vis = self.cnn(obs["visual"])
        vec = self.vec_net(obs["vector"])
        return self.fusion(torch.cat([vis, vec], dim=1))


# ═══════════════════════════════════════════════════════════════
# Env Factory
# ═══════════════════════════════════════════════════════════════

def make_env(level_config, seed=None):
    def _init():
        cfg = copy.deepcopy(level_config)
        env = MazeEnv(config=cfg)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def make_vec_env(level_config, n_envs=N_ENVS, normalize=True):
    envs = DummyVecEnv([make_env(level_config, seed=i * 1000) for i in range(n_envs)])
    if normalize:
        envs = VecNormalize(
            envs,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=0.99,
        )
    return envs


# ═══════════════════════════════════════════════════════════════
# Training Callback
# ═══════════════════════════════════════════════════════════════

class LogCallback(BaseCallback):
    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.ep_rewards = []
        self.ep_wins = []
        self._running_rewards = None

    def _on_training_start(self):
        self._running_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [])
        self._running_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(float(self._running_rewards[i]))
                self._running_rewards[i] = 0.0
                if i < len(infos) and "win" in infos[i]:
                    self.ep_wins.append(1.0 if infos[i]["win"] else 0.0)

        if self.n_calls % self.log_interval == 0 and self.ep_rewards:
            recent_r = self.ep_rewards[-100:]
            recent_w = self.ep_wins[-100:]
            mean_r = np.mean(recent_r)
            win_rate = np.mean(recent_w) if recent_w else 0.0
            print(
                f"  step={self.num_timesteps:>8d} | "
                f"mean_r={mean_r:>7.2f} | "
                f"win_rate={win_rate:.2%} | "
                f"episodes={len(self.ep_rewards)}"
            )
        return True


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(model, level_config, n_episodes=20, seeds=None):
    """Evaluate model on a level. Returns (mean_reward, success_rate)."""
    if seeds is None:
        seeds = list(range(9000, 9000 + n_episodes))

    rewards = []
    wins = []

    for seed in seeds[:n_episodes]:
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
# Build Model
# ═══════════════════════════════════════════════════════════════

def build_model(env):
    policy_kwargs = dict(
        features_extractor_class=SmallDictExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=nn.ReLU,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        verbose=0,
    )
    return model


# ═══════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════

def train():
    print(f"Device: {DEVICE}")
    print(f"Parallel envs: {N_ENVS}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Curriculum: {len(CURRICULUM_PLAN)} levels")
    print("=" * 60)

    first_cfg = copy.deepcopy(CURRICULUM[CURRICULUM_PLAN[0]["level"]])
    vec_env = make_vec_env(first_cfg)
    model = build_model(vec_env)

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model params: {total_params:,}")
    print("=" * 60)

    for stage_idx, stage in enumerate(CURRICULUM_PLAN):
        level = stage["level"]
        timesteps = stage["timesteps"]
        mastery_threshold = stage["mastery"]
        level_cfg = copy.deepcopy(CURRICULUM[level])
        level_name = level_cfg.get("name", f"Level {level}")

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx+1}/{len(CURRICULUM_PLAN)}: Level {level} - {level_name}")
        print(f"Timesteps: {timesteps:,} | Mastery gate: {mastery_threshold:.0%}")
        print(f"Map: {level_cfg['width']}x{level_cfg['height']} | "
              f"Floors: {level_cfg['n_floors']} | "
              f"Enemies: {len(level_cfg.get('enemies', []))} | "
              f"Locks: {len(level_cfg.get('locks', []))}")
        print("-" * 60)

        vec_env = make_vec_env(level_cfg)
        model.set_env(vec_env)

        callback = LogCallback(log_interval=10000)
        t0 = time.time()
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )
        elapsed = time.time() - t0

        mean_r, win_rate = evaluate(model, level_cfg, n_episodes=30)
        print(f"\nLevel {level} done in {elapsed:.0f}s")
        print(f"  Eval: mean_reward={mean_r:.2f} | success_rate={win_rate:.1%}")

        ckpt_path = os.path.join(OUTPUT_DIR, f"ppo_level_{level}.zip")
        model.save(ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        if win_rate < mastery_threshold:
            print(f"  Below mastery ({win_rate:.1%} < {mastery_threshold:.0%})")
            extra = min(timesteps, 200_000)
            print(f"  Training {extra:,} more steps...")
            callback2 = LogCallback(log_interval=10000)
            model.learn(
                total_timesteps=extra,
                callback=callback2,
                reset_num_timesteps=False,
            )
            mean_r2, win_rate2 = evaluate(model, level_cfg, n_episodes=30)
            print(f"  Retry eval: mean_reward={mean_r2:.2f} | success_rate={win_rate2:.1%}")
            model.save(ckpt_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nFinal evaluation across all levels:")
    for stage in CURRICULUM_PLAN:
        level = stage["level"]
        cfg = copy.deepcopy(CURRICULUM[level])
        mean_r, win_rate = evaluate(model, cfg, n_episodes=30)
        print(f"  Level {level:>2d} ({cfg['name']:<15s}) | "
              f"reward={mean_r:>7.2f} | success={win_rate:.1%}")

    final_path = os.path.join(OUTPUT_DIR, "ppo_final.zip")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")


if __name__ == "__main__":
    train()
