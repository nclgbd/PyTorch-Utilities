
::conda activate torch_base
pip uninstall pytorch_vision_utils -y
::bumpversion --dry-run --allow-dirty patch
py -m build
pip install dist/pytorch_vision_utils-0.3.6.tar.gz
