import os
import torch
from torch import nn

from pytorch_vision_utils.Utilities import TrainingUtilities


# Default directory names
cwd = os.getcwd()

TEST_DIR = str(os.path.join(cwd, "test_data"))
MODEL_DIR = str(os.path.join(cwd, "saved_models"))
MEDIA_DIR = str(os.path.join(cwd, 'media'))
INC_DIR = str(os.path.join(cwd, 'incorrect_images'))
MODEL_NAME = "mobilenetv2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="mobilenetv2", device=device)


def run_epoch():
    results = tuple() # empty tuple
            
    params = {"criterion": nn.CrossEntropyLoss(),
              "optimizer": torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta),
              "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta), mode='min', 
                                                                         factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)}
    
    results = train_utils.train(model_name=MODEL_NAME, model_path=MODEL_DIR, inc_path=INC_DIR, media_dir=MEDIA_DIR, show_graphs=False, 
                                dry_run=True, debug=True, max_epoch=2)
    
    return -1 if len(results) == 0 else 0


def test_run_epoch():
    assert run_epoch() == 0