#!/usr/bin/env bash
set -euo pipefail
python -m clashrl train \
  --run runs/v31 \
  --updates 100 \
  --rollout-steps 8192 \
  --num-envs 24 \
  --workers 0 \
  --snapshot-every 5 \
  --tournament-every-games 100 \
  --tournament-games-per-pair 2 \
  --tournament-max-models 6 \
  --visualize-every 0
