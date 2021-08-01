import os

os.system("conda env update --name torch_base --file conda-envs/torch_base.yml")


from . import avail_custom_models
from .avail_custom_models import mobilenetv2
from .avail_custom_models import xception
from .avail_custom_models import avail_models

from .mobilenetv2 import mobilenetv2, MobileNetV2
from .xception import xception, Xception

version='0.3.2'

print("Available models:\n", avail_models)
