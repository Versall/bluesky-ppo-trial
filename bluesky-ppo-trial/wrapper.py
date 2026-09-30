# wrappers.py
"""
Wrapper & util untuk mengekstrak metrik kompetisi dari info env.

BlueSky-Gym mengembalikan info dict dengan key metrik (nama bisa bervariasi
antar versi). Fungsi di sini melakukan pencocokan substring agar robust.
"""
from __future__ import annotations

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

from config import METRIC_KEYS


def _match_keys(info: Dict[str, Any], wanted: List[str]) -> Dict[str, Any]:
    """Cocokkan key info dengan daftar yang diinginkan via substring."""
    out: Dict[str, Any] = {}
    lowered = {k.lower(): k for k in info.keys()}
    for w in wanted:
        wl = w.lower()
        # exact dulu
        if wl in lowered:
            out[w] = info[lowered[wl]]
            continue
        # substring
        for lk, orig in lowered.items():
            if wl in lk:
                out[w] = info[orig]
                break
    return out


def _to_scalar(v: Any) -> float:
    """Konversi nilai metrik ke scalar float bila memungkinkan."""
    if isinstance(v, (int, float, np.integer, np.floating, bool)):
        return float(v)
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return float(v.reshape(-1)[0])
        return float(np.sum(v))
    if isinstance(v, (list, tuple)):
        try:
            return float(np.sum(v))
        except Exception:
            return float("nan")
    return float("nan")


class MetricExtractorWrapper(Wrapper):
    """
    Menyimpan metrik terakhir yang muncul di info ke self.last_metrics.
    Asumsi: metrik di-publish di info pada step terakhir episode (done=True)
    atau di setiap step (kita overwrite, yang terakhir menang).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_metrics: Dict[str, float] = {}

    def reset(self, **kwargs):
        self.last_metrics = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extracted = _match_keys(info, METRIC_KEYS)
        if extracted:
            self.last_metrics = {k: _to_scalar(v) for k, v in extracted.items()}
        return obs, reward, terminated, truncated, info