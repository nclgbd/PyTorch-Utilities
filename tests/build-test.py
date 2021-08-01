import os
import sys
import torch

from pytorch_vision_utils.Utilities import build
from pytorch_vision_utils.Utilities import clear_dirs, time_to_predict, DataVisualizationUtilities, TrainingUtilities
from pretrainedmodels.models.xception import Xception
from pretrainedmodels.models.mobilenetv2 import MobileNetV2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


# DIRECTORY NAMES
cwd = os.getcwd()
TEST_DIR = str(os.path.join(cwd, "test_data"))
MODEL_DIR = str(os.path.join(cwd, "saved_models"))
MEDIA_DIR = str(os.path.join(cwd, 'media'))
INC_DIR = str(os.path.join(cwd, 'incorrect_images'))

TrainingUtilities(data_dir=TEST_DIR, model_name="mobilenetv2")
TrainingUtilities(data_dir=TEST_DIR, model_name="xception")

sys.exit(0)