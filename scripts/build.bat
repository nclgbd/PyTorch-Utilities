@RD /S /Q ".\dist"
pip uninstall pytorch_vision_utils -y
py -m build
pip install dist/pytorch_vision_utils-0.4.2.tar.gz
