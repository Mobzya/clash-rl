#!/usr/bin/env bash
set -euo pipefail
python -m clashrl benchmark --workers 0 2 4 auto --num-envs "${1:-24}" --steps 2048 --device cpu
