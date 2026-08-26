#!/usr/bin/env bash
set -euo pipefail
python -m clashrl dashboard --run "${1:-runs/v31}"
