#!/bin/bash

echo "$CONDA/bin" >> "$GITHUB_PATH"
conda activate pytorch_vision_dev
python setup.py sdist bdist_wheel
twine upload dist/pytorch_vision_utils-0.3.9.tar.gz