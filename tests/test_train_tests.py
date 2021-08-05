import os
import sys
import torch
import zipfile
from mdutils.mdutils import MdUtils

from pytorch_vision_utils.Utilities import TrainingUtilities
from pytorch_vision_utils.Utilities import clear_dirs, build


# Default directory names
cwd = os.getcwd()
# build("parameters.json")
TEST_DIR = str(os.path.join(cwd, "test_data"))
MODEL_DIR = str(os.path.join(cwd, "saved_models"))
MEDIA_DIR = str(os.path.join(cwd, 'media'))
INC_DIR = str(os.path.join(cwd, 'incorrect_images'))
MODEL_NAME = "mobilenetv2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="mobilenetv2")


def run_epoch():
    results = tuple() # empty tuple
    results = train_utils.train(model_name=MODEL_NAME, model_path=MODEL_DIR, inc_path=INC_DIR, media_dir=MEDIA_DIR, show_graphs=False, dry_run=True, debug=True, max_epoch=2)
    
    return -1 if len(results) == 0 else 0


def test_run_epoch():
    assert run_epoch() == 0