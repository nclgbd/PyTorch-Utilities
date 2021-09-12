import os
import json
import sys
import torch
import zipfile

from mdutils.mdutils import MdUtils

from pytorch_vision_utils.Utilities import TrainingUtilities
from pytorch_vision_utils.Utilities import clear_dirs, build
from torch import nn


# Default directory names
with open("parameters.json", "r") as f:
    print("Loading parameters...")
    params = dict(json.load(f))
    
    DATA_DIR = params["DATA_DIR"]
    TEST_DIR = params["TEST_DIR"]
    MODEL_DIR = params["MODEL_DIR"]
    MEDIA_DIR = params["MEDIA_DIR"]
    INC_DIR = params["INC_DIR"]
    
    print("Loading parameters complete!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_mobilenetv2_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="mobilenetv2")
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, MODEL_DIR, criterion, optimizer, fold+1, ascii_=True, scheduler=lr_scheduler, 
                            dry_run=False, show_graphs=False, inc_path=INC_DIR, max_epoch=1)
    return 0 if res else -1


def run_xception_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="xception")
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, MODEL_DIR, criterion, optimizer, fold+1, ascii_=True, scheduler=lr_scheduler, 
                            dry_run=False, show_graphs=False, inc_path=INC_DIR, max_epoch=1)
    return 0 if res else -3


def run_vggm_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="vggm")
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, MODEL_DIR, criterion, optimizer, fold+1, ascii_=True, scheduler=lr_scheduler, 
                            dry_run=False, show_graphs=False, inc_path=INC_DIR, max_epoch=1)
    return 0 if res else -4


def run_resnext101_32x4d_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="resnext101_32x4d", debug=True)
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, MODEL_DIR, criterion, optimizer, fold+1, ascii_=True, scheduler=lr_scheduler, 
                            dry_run=False, show_graphs=False, inc_path=INC_DIR, max_epoch=1)
    return 0 if res else -5


def run_inceptionv4_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="inceptionv4", debug=True)
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, MODEL_DIR, criterion, optimizer, fold+1, ascii_=True, scheduler=lr_scheduler, 
                            dry_run=False, show_graphs=False, inc_path=INC_DIR, max_epoch=1)
    return 0 if res else -5


def test_run_mobilenetv2_epoch():
    train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="mobilenetv2", 
                                    parameters_path="test_params.json")
    assert run_mobilenetv2_epoch(train_utils) == 0
    

def test_run_xception_epoch():
    train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="xception", 
                                    parameters_path="test_params.json")
    assert run_xception_epoch(train_utils) == 0 
    

def test_run_vggm_epoch():
    train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="vggm", 
                                    parameters_path="test_params.json")
    assert run_vggm_epoch(train_utils) == 0 
   

def test_run_resnext101_32x4d_epoch():
    train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="resnext101_32x4d", 
                                    parameters_path="test_params.json")
    assert run_resnext101_32x4d_epoch(train_utils) == 0
    

def test_run_inceptionv4_epoch():
    train_utils = TrainingUtilities(data_dir=TEST_DIR, model_dir=MODEL_DIR, model_name="inceptionv4", 
                                    parameters_path="test_params.json")
    assert run_inceptionv4_epoch(train_utils) == 0
    