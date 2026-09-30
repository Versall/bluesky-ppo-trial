# config.py
from pathlib import Path

# ---------- Reproducibility ----------
SEED = 42

# ---------- Paths ----------
RUN_DIR = Path("runs/ppo_trial1")
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUN_DIR / "model.zip"
VECNORM_PATH = RUN_DIR / "vecnormalize.pkl"
TB_LOG_DIR = str(RUN_DIR / "tb")
EVAL_CSV = RUN_DIR / "eval.csv"

# ---------- Env ----------
ENV_ID = "CompetitionEnv-v0"
N_ENV_TRAIN = 4          # paralel env untuk training
N_ENV_EVAL = 1

# ---------- Training ----------
TOTAL_TIMESTEPS = 500_000

# ---------- PPO hyperparameters ----------
PPO_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=2048,        # per env; total rollout = n_steps * N_ENV_TRAIN
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    target_kl=0.05,      # safety: hindari update terlalu besar
    policy_kwargs=dict(net_arch=[256, 256]),
)

# ---------- Evaluation ----------
EVAL_EPISODES = 50       # naikkan ke 1000 setelah stabil
EVAL_SEED = 42           # seed resmi kompetisi

# ---------- Normalization ----------
NORM_OBS = True
NORM_REWARD = True
CLIP_OBS = 10.0
CLIP_REWARD = 10.0

# ---------- Metrics yang diharapkan ----------
# Substring pencocokan; order penting (yang lebih spesifik di atas)
METRIC_KEYS = [
    "waypoint_reached",
    "intrusion_events",
    "intrusion_time",
    "restricted_area_events",
    "time_in_restricted_area",
    "sector_exit_events",
    "time_outside_sector",
    "flight_time",
    "total_reward",
]