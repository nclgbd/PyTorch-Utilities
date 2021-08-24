#!/bin/bash --login

ls -la
mkdir -p media/ incorrect_images/ saved_models/

conda init bash

bash ~/.bashrc
bash /opt/conda/etc/profile.d/conda.sh

conda activate pytorch_vision_dev
pip install -e .