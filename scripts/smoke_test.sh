#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src"
export MPLCONFIGDIR="$REPO_DIR/results/smoke/.matplotlib"

cd "$REPO_DIR"
python3 -m compileall -q src tests scripts
python3 -m unittest discover -s tests -v
python3 scripts/run_experiment.py --config configs/smoke.yaml
python3 scripts/validate_outputs.py \
  --output results/smoke \
  --expected-metric-rows 576 \
  --expected-methods \
    gradient_boosting \
    knn \
    mean_mode \
    plain_vae \
    plain_vae_structured \
    random_forest \
    sa_vae \
    sa_vae_structured
echo "SMOKE_TEST_PASSED"
