#!/bin/bash

python setup.py sdist bdist_wheel
twine upload dist/pytorch_vision_utils-0.4.0.tar.gz
