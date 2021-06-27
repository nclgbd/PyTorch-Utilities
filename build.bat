::conda activate torch_base

pip uninstall pretrained-models.pytorch

Remove-Item -LiteralPath "./pretrained-models.pytorch" -Force -Recurse
git clone https://github.com/nclgbd/pretrained-models.pytorch.git
cd pretrained-models.pytorch
python setup.py install
pip install -r requirements.txt
cd ..

pip uninstall pytorch_vision_utils -y
py -m build
pip install dist/pytorch_vision_utils-0.2.1.tar.gz

::py -m pip install --upgrade twine
::py -m twine upload dist/* --verbose