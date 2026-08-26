#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-runs/v31}"
python -m clashrl tournament --run "$RUN" --games-per-pair 2 --max-models 6
python -m clashrl leaderboard --run "$RUN" --current
