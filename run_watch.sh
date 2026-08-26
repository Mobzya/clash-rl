#!/usr/bin/env bash
set -euo pipefail
RUN="${1:-runs/v31}"
if [ ! -f "$RUN/latest.pt" ]; then
  python -m clashrl init --run "$RUN"
fi
python -m clashrl watch --run "$RUN" --a latest --b random --speed 4 --draft
