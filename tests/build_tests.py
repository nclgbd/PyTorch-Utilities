import os
import sys
import torch

def build_test():
    print("Importing custom packages...")
    from pytorch_vision_utils.Utilities import DataVisualizationUtilities, TrainingUtilities
    from pytorch_vision_utils.Utilities import clear_dirs, time_to_predict
    from pretrainedmodels.models.xception import Xception
    from pretrainedmodels.models.mobilenetv2 import MobileNetV2


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)


    # DIRECTORY NAMES
    print("Creating folders...")
    cwd = os.getcwd()
    TEST_DIR = str(os.path.join(cwd, "test_data"))
    MODEL_DIR = str(os.path.join(cwd, "saved_models"))
    MEDIA_DIR = str(os.path.join(cwd, 'media'))
    INC_DIR = str(os.path.join(cwd, 'incorrect_images'))


    print("Initializing TrainingUtilities class...")
    TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="mobilenetv2")
    TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="xception")

    print("Initializing DatavisualizationUtilities class...")
    DataVisualizationUtilities()
    
    return 0