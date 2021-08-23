#!/bin/bash --login

ls -la
mkdir -p media/ incorrect_images/ saved_models/

set -euo pipefail
conda activate pytorch_vision_dev
pip install -e .