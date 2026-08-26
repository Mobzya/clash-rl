#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-runs/v31}"
python -m clashrl dashboard --run "$RUN" &
DASH_PID=$!
cleanup() { kill "$DASH_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
python -m clashrl train \
  --run "$RUN" \
  --updates 100 \
  --rollout-steps 8192 \
  --num-envs 24 \
  --workers 0 \
  --snapshot-every 5 \
  --tournament-every-games 100 \
  --tournament-games-per-pair 2 \
  --tournament-max-models 6 \
  --visualize-every 0
