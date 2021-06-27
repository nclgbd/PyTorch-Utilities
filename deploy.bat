py -m pip install --upgrade twine
py -m twine upload --repository testpypi dist/*

::pip install dist/pytorch_vision_utils-0.2.1.tar.gz