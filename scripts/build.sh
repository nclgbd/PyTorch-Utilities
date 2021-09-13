#!/bin/bash

mkdir -p /workdir/media/ /workdir/incorrect_images/ /workdir/saved_models/ /workdir/test_data/ /workdir/data/
conda activate pytorch_vision_dev
pip install -e .
