@RD /S /Q ".\dist"
pip uninstall pytorch_vision_utils -y
py -m build
pip install dist/pytorch_vision_utils-0.3.13.tar.gz
