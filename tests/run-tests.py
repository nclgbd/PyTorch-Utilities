from subprocess import call


print("Running all tests...\n")

test_names = ["build-test"]
for name in test_names:
    print("Running:", name)
    assert call('python tests/{}.py'.format(name).split()) == 0
    