#!/bin/bash

echo "$CONDA/bin" >> "$GITHUB_PATH"
conda activate pytorch_vision_dev
ls -la
pip install -e .
mkdir -p media/ incorrect_images/ saved_models/
pytest -v -s
echo "Printing contents of local dir"
ls -la
echo "Printing contents of media/"
ls -la media/
cat media/report_mobilenetv2.md