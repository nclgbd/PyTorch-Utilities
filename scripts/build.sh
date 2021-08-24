#!/bin/bash --login

ls -la
mkdir -p media/ incorrect_images/ saved_models/

conda init bash

export -f conda
export -f __conda_activate
export -f __conda_reactivate
export -f __conda_hashr

bash ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_vision_dev
pip install -e .