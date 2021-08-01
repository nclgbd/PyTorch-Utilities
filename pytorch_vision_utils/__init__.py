import os

os.system("conda activate torch_base")

from . import custom_models
from .custom_models import mobilenetv2, MobileNetV2
from .custom_models import xception, Xception

version='0.3.2'
