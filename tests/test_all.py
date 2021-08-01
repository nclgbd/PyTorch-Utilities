from build_tests import build_test
from subprocess import call


print("Running all tests...\n")

tests = [build_test]
for test in tests:
    print("Running:", test.__name__)
    assert test() == 0
    