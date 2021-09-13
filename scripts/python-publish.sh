#!/bin/bash
python setup.py sdist bdist_wheel
twine upload dist/pytorch_vision_utils-0.3.15.tar.gz