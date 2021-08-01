import os


print("Running all tests...\n")

test_names = ["build-test"]
for name in test_names:
    print("Running:", name)
    os.system('python tests/{}.py'.format(name))
    