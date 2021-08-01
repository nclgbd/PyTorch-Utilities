# from pytorch_vision_utils import pytorch_vision_utils
from pytorch_vision_utils import Utilities
from pytorch_vision_utils.Utilities import DataVisualizationUtilities, TrainingUtilities
from pytorch_vision_utils.Utilities import clear_dirs, time_to_predict

from build_tests import build_test
from subprocess import call


build_tests = [build_test]
all_tests = [build_tests]
for tests in all_tests:
    for test in tests:
        print("Running:", test.__name__)
        assert test() == 0
    