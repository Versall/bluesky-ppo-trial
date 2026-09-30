# evaluate.py
"""
Evaluasi model terlatih pada N episode, menghasilkan CSV metrik per-episode.
Mengikuti protokol kompetisi: seed 42, distribusi skenario default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bluesky_gym  # noqa: F401
from config import (
    ENV_ID, EVAL_CSV, EVAL_EPISODES, EVAL_SEED, METRIC_KEYS,
    MODEL_PATH, VECNORM_PATH,
)
from wrappers import MetricExtractorWrapper, _match_keys, _to_scalar

import bluesky_gym
bluesky_gym.register_envs()   # ← WAJIB
env = gym.make('CompetitionEnv-v0')

def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID)
        env = FlattenObservation(env)
        env = MetricExtractorWrapper(env)
        return env
    return _init


def evaluate(
    model_path: Path,
    vecnorm_path: Path,
    n_episodes: int,
    seed: int,
    out_csv: Path,
) -> pd.DataFrame:
    vec_env = DummyVecEnv([make_env(seed)])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(str(model_path), env=vec_env, device="auto")

    print(f"[eval] Menjalankan {n_episodes} episode (seed={seed})...")
    records: List[Dict[str, float]] = []

    obs = vec_env.reset()
    ep_idx = 0
    ep_reward = 0.0
    ep_len = 0

    while ep_idx < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        ep_reward += float(reward[0])
        ep_len += 1

        if done[0]:
            # info[0] bisa berisi EpisodeStatistics dan metrik kompetisi.
            # Kalau metrik ada di env wrapper (bukan info), ambil dari sana:
            raw_info = info[0] if isinstance(info, (list, tuple)) else info

            # Coba ambil dari info dulu
            metrics = _match_keys(raw_info, METRIC_KEYS)
            metrics = {k: _to_scalar(v) for k, v in metrics.items()}

            # Fallback: ambil dari wrapper terakhir (env.env.env...)
            if not metrics:
                try:
                    inner = vec_env.envs[0]
                    while hasattr(inner, "env"):
                        if isinstance(inner, MetricExtractorWrapper):
                            metrics = dict(inner.last_metrics)
                            break
                        inner = inner.env
                except Exception:
                    pass

            row = {"episode": ep_idx, "ep_reward": ep_reward, "ep_len": ep_len}
            row.update(metrics)
            records.append(row)

            ep_idx += 1
            ep_reward = 0.0
            ep_len = 0
            obs = vec_env.reset()

            if (ep_idx % 10) == 0 or ep_idx == n_episodes:
                print(f"[eval] {ep_idx}/{n_episodes} episode selesai")

    df = pd.DataFrame(records)

    # Ringkasan
    print("\n=== RINGKASAN EVALUASI ===")
    summary_cols = [c for c in df.columns if c not in ("episode",)]
    print(df[summary_cols].describe().T[["mean", "std", "min", "max"]])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[eval] CSV disimpan: {out_csv}")
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--vecnorm", type=Path, default=VECNORM_PATH)
    p.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--out", type=Path, default=EVAL_CSV)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes,
        seed=args.seed,
        out_csv=args.out,
    )