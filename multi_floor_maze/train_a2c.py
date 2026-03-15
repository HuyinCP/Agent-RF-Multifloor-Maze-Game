"""
A2C Baseline Training — same curriculum, same SmallDictExtractor as PPO.
Usage: python train_a2c.py
"""
import copy, os, time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

from mfm.env import MazeEnv, CURRICULUM

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
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        vis = self.cnn(obs["visual"])
        vec = self.vec_net(obs["vector"])
        return self.fusion(torch.cat([vis, vec], dim=1))


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
            print(f"  step={self.num_timesteps:>8d} | "
                  f"mean_r={np.mean(recent_r):>7.2f} | "
                  f"win={np.mean(recent_w) if recent_w else 0:.2%}")
        return True


def make_env(cfg, seed=None):
    def _init():
        e = MazeEnv(config=copy.deepcopy(cfg))
        if seed is not None:
            e.reset(seed=seed)
        return e
    return _init


def evaluate(model, cfg, n_episodes=30):
    rewards, wins = [], []
    for seed in range(9000, 9000 + n_episodes):
        e = MazeEnv(config=copy.deepcopy(cfg))
        obs, _ = e.reset(seed=seed)
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = e.step(int(action))
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        wins.append(1.0 if info.get("win", False) else 0.0)
    return float(np.mean(rewards)), float(np.mean(wins))


def train():
    print(f"A2C Baseline Training")
    print(f"Device: {DEVICE} | Envs: {N_ENVS}")
    print("=" * 60)

    first_cfg = copy.deepcopy(CURRICULUM[1])
    vec_env = DummyVecEnv([make_env(first_cfg, seed=i*1000) for i in range(N_ENVS)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)

    model = A2C(
        "MultiInputPolicy", vec_env,
        learning_rate=7e-4,
        n_steps=16,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            features_extractor_class=SmallDictExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
            activation_fn=nn.ReLU,
        ),
        device=DEVICE, verbose=0,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Params: {total_params:,}")
    print("=" * 60)

    for stage in CURRICULUM_PLAN:
        level = stage["level"]
        ts = stage["timesteps"]
        cfg = copy.deepcopy(CURRICULUM[level])

        print(f"\nLevel {level}: {cfg['name']} — {ts:,} steps")
        vec_env = DummyVecEnv([make_env(cfg, seed=i*1000) for i in range(N_ENVS)])
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
        model.set_env(vec_env)

        t0 = time.time()
        model.learn(total_timesteps=ts, callback=LogCallback(), reset_num_timesteps=False)
        elapsed = time.time() - t0

        mean_r, win_rate = evaluate(model, cfg)
        print(f"  Done in {elapsed:.0f}s | reward={mean_r:.2f} | win={win_rate:.1%}")

        path = os.path.join(OUTPUT_DIR, f"a2c_level_{level}.zip")
        model.save(path)
        print(f"  Saved: {path}")

        if win_rate < stage["mastery"]:
            print(f"  Below mastery ({win_rate:.1%} < {stage['mastery']:.0%}), extra 200K...")
            model.learn(total_timesteps=200_000, callback=LogCallback(), reset_num_timesteps=False)
            mean_r2, wr2 = evaluate(model, cfg)
            print(f"  Retry: reward={mean_r2:.2f} | win={wr2:.1%}")
            model.save(path)

    final = os.path.join(OUTPUT_DIR, "a2c_final.zip")
    model.save(final)
    print(f"\n{'='*60}")
    print(f"A2C TRAINING COMPLETE — {final}")

    print("\nFinal eval:")
    for stage in CURRICULUM_PLAN:
        level = stage["level"]
        cfg = copy.deepcopy(CURRICULUM[level])
        mr, wr = evaluate(model, cfg)
        print(f"  L{level:>2d} {cfg['name']:<12s} | reward={mr:>7.2f} | win={wr:.1%}")


if __name__ == "__main__":
    train()
