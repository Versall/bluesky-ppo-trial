#!/usr/bin/env bash
# run_all.sh — jalankan smoke test -> training -> evaluasi
set -euo pipefail

echo "=========================================="
echo "[1/3] Smoke test"
echo "=========================================="
python -m smoke_test

echo
echo "=========================================="
echo "[2/3] Training PPO"
echo "=========================================="
python -m train

echo
echo "=========================================="
echo "[3/3] Evaluasi"
echo "=========================================="
python -m evaluate --episodes 50

echo
echo "Selesai. Lihat hasil di: runs/ppo_trial1/"