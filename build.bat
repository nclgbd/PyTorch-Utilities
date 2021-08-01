
::conda activate torch_base
pip uninstall pytorch_vision_utils -y
py -m build
pip install dist/pytorch_vision_utils-0.3.2.tar.gz

::py -m pip install --upgrade twine
::py -m twine upload dist/* --verbose