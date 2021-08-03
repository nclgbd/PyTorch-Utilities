# from pytorch_vision_utils import pytorch_vision_utils
from pytorch_vision_utils import Utilities
from pytorch_vision_utils.Utilities import DataVisualizationUtilities, TrainingUtilities
from pytorch_vision_utils.Utilities import clear_dirs, time_to_predict

from build_tests import build_test
from train_tests import run_epoch


build_tests = [build_test]
train_tests = [run_epoch]

all_tests = [build_tests, train_tests]
for tests in all_tests:
    for test in tests:
        print("Running:", test.__name__)
        assert test() == 0
    