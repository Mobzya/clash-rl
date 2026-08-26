#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "== Clash RL v3.1 verification =="
python -m clashrl doctor
python -m compileall -q clashrl tests
python -m unittest discover -s tests -v
python -m clashrl smoke --steps 800

TMP_RUN="$(mktemp -d -t clashrl-v31-verify-XXXXXX)"
cleanup() { rm -rf "$TMP_RUN"; }
trap cleanup EXIT INT TERM

python -m clashrl train \
  --run "$TMP_RUN" \
  --updates 1 \
  --rollout-steps 512 \
  --num-envs 4 \
  --workers 0 \
  --epochs 1 \
  --minibatch 128 \
  --snapshot-every 1 \
  --no-tournament \
  --visualize-every 0 \
  --device cpu

echo "VERIFY PASS"
echo "Next: ./run_demo.sh  or  ./run_train.sh"
