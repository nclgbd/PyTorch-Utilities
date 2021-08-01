
pip uninstall pretrainedmodels -y
git clone https://github.com/nclgbd/pretrained-models.pytorch.git
cd pretrained-models.pytorch
git pull origin master
python setup.py install
pip install -r requirements.txt
cd ..

pip uninstall pytorch_vision_utils -y
::bumpversion patch
py -m build
pip install dist/pytorch_vision_utils-0.3.2.tar.gz

py -m pip install --upgrade twine
py -m twine upload dist/pytorch_vision_utils-0.3.2.tar.gz --verbose
