import os
import sys
import torch

from pytorch_vision_utils.Utilities import DataVisualizationUtilities, TrainingUtilities


def build_test():
    # Default directory names
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



def test_build_tests():
    assert build_test() == 0
