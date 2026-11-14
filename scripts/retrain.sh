#!/usr/bin/env bash
# Retraining job run by the trainer service (and usable by hand).
#   1. import user-labelled field photos into data/raw
#   2. if there were new photos -> `dvc add data/raw` = new data version
#   3. rerun the full DVC pipeline (ingest ... evaluate + registry gate)
#   4. push data/models to the DVC remote
# After it finishes, commit data/raw.dvc + dvc.lock to record the new version in Git.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.data.import_feedback
new=$(python -c 'import json; print(json.load(open("reports/feedback_import.json")).get("imported", 0))')
if [ "$new" -gt 0 ]; then
  echo "Running stage 'version-data': $new new labelled photos -> dvc add data/raw"
  dvc add data/raw
fi

dvc repro -f evaluate
dvc push || echo "WARNING: dvc push failed (remote not reachable?)"
