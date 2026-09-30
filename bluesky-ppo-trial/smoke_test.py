# smoke_test.py
"""
Verifikasi cepat: env bisa dibuat, reset, dan step manual tanpa error.
Jalankan SEBELUM training.
"""
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import bluesky_gym  # noqa: F401  -> registrasi env
from config import ENV_ID, SEED
from wrappers import MetricExtractorWrapper


def main():
    print(f"[smoke] Membuat env: {ENV_ID}")
    env = gym.make(ENV_ID)
    env = FlattenObservation(env)
    env = MetricExtractorWrapper(env)

    print(f"[smoke] observation_space: {env.observation_space}")
    print(f"[smoke] action_space     : {env.action_space}")

    obs, info = env.reset(seed=SEED)
    print(f"[smoke] obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

    total_r = 0.0
    for t in range(20):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if term or trunc:
            print(f"[smoke] Episode selesai di step {t}, total reward {total_r:.3f}")
            obs, info = env.reset()
            total_r = 0.0

    print("[smoke] Metrik terakhir yang terekstrak:")
    for k, v in env.last_metrics.items():
        print(f"   {k:30s} = {v}")

    print("[smoke] OK.")


if __name__ == "__main__":
    main()