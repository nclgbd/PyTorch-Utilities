import os
from pytorch_vision_utils.DataVisualizations import DataVisualizationUtilities
from pytorch_vision_utils.TrainingUtilities import TrainingUtilities


def build_test():
    # Default directory names
    print("Checking for correct folders...")
    cwd = os.getcwd()
    DATA_DIR = str(os.path.join(cwd, "data"))
    TEST_DIR = str(os.path.join(cwd, "test_data"))
    MODEL_DIR = str(os.path.join(cwd, "saved_models"))
    MEDIA_DIR = str(os.path.join(cwd, 'media'))
    INC_DIR = str(os.path.join(cwd, 'incorrect_images'))

    
    dirs = set(os.listdir(cwd))
    print(dirs)
    
    if "data" not in dirs:
        print("Missing data/ directory")
        return -1
    
    if "test_data" not in dirs:
        print("Missing test_data/ directory")
        return -2
    
    if "saved_models" not in dirs:
        print("Missing saved_models/ directory")
        return -3
    
    if "media" not in dirs:
        print("Missing media/ directory")
        return -4
    
    if "incorrect_images" not in dirs:
        print("Missing incorrect_images/ directory")
        return -5

    print("Initializing TrainingUtilities class...")
    TrainingUtilities(model_name="mobilenetv2", parameters_path="test_parameters.yml")
    TrainingUtilities(model_name="xception", parameters_path="test_parameters.yml")
    TrainingUtilities(model_name="resnext101_32x4d", parameters_path="test_parameters.yml")
    TrainingUtilities(model_name="vggm", parameters_path="test_parameters.yml")
    TrainingUtilities(model_name="inceptionv4", parameters_path="test_parameters.yml")
   

    print("Initializing DatavisualizationUtilities class...")
    DataVisualizationUtilities()

    return 0


def test_build_tests():
    assert build_test() == 0
