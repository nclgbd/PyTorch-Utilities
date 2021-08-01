# from pytorch_vision_utils import pytorch_vision_utils
from build_tests import build_test, import_test
from subprocess import call


build_tests = [import_test, build_test]
all_tests = [build_tests]
for tests in all_tests:
    for test in tests:
        print("Running:", test.__name__)
        assert test() == 0
    