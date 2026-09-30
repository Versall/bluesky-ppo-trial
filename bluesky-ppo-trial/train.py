# train.py
"""
Training PPO untuk CompetitionEnv-v0 (single-agent).
Menyimpan model, VecNormalize statistics, dan log TensorBoard.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bluesky_gym  # noqa: F401
from config import (
    CLIP_OBS, CLIP_REWARD, ENV_ID, MODEL_PATH, N_ENV_TRAIN, NORM_OBS, NORM_REWARD,
    PPO_CONFIG, RUN_DIR, SEED, TB_LOG_DIR, TOTAL_TIMESTEPS, VECNORM_PATH,
)
from wrappers import MetricExtractorWrapper


def make_env(rank: int, seed: int = SEED):
    """Factory: buat env tunggal, seed, dan bungkus."""
    def _init():
        env = gym.make(ENV_ID)
        env = FlattenObservation(env)
        env = MetricExtractorWrapper(env)
        env = Monitor(env)                      # untuk episode stats
        env = RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def main():
    print(f"[train] Membuat {N_ENV_TRAIN} env paralel...")
    vec_env = DummyVecEnv([make_env(i) for i in range(N_ENV_TRAIN)])

    # Normalisasi obs & reward (sangat direkomendasikan untuk SB3)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=NORM_OBS,
        norm_reward=NORM_REWARD,
        clip_obs=CLIP_OBS,
        clip_reward=CLIP_REWARD,
    )

    # Checkpoint tiap 50k step
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENV_TRAIN, 1),
        save_path=str(RUN_DIR / "checkpoints"),
        name_prefix="ppo",
    )

    print("[train] Inisialisasi PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        seed=SEED,
        tensorboard_log=TB_LOG_DIR,
        **PPO_CONFIG,
    )

    print(f"[train] Mulai training {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ckpt_cb,
        progress_bar=True,
        tb_log_name="ppo_trial1",
    )

    print("[train] Menyimpan model & VecNormalize stats...")
    model.save(str(MODEL_PATH))
    vec_env.save(str(VECNORM_PATH))
    print(f"[train] Selesai. Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()