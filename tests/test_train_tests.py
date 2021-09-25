import os
import json
import torch
from pytorch_vision_utils.Utilities import TrainingUtilities
from torch import nn

# Default directory names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_mobilenetv2_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="mobilenetv2", debug=True)
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, criterion, optimizer, fold+1, scheduler=lr_scheduler, 
                            dry_run=True, show_graphs=False, max_epoch=1)
    return 0 if res else -1


def run_xception_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="xception", debug=True)
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, criterion, optimizer, fold+1, scheduler=lr_scheduler, 
                            dry_run=True, show_graphs=False, max_epoch=1)
    return 0 if res else -1


def run_vggm_epoch(train_utils):
    fold = 0
    train_utils.set_model_parameters(model_name="vggm", debug=True)
    train_idx, test_idx = train_utils.dataset.folds[fold]
    train_dataset = torch.utils.data.Subset(train_utils.dataset, train_idx)
    test_dataset = torch.utils.data.Subset(train_utils.dataset, test_idx)
    
    train_dataset.transform = train_utils.train_transform
    test_dataset.transform = train_utils.test_transform
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(train_utils.model.parameters(), lr=train_utils.eta)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=train_utils.factor, patience=train_utils.lr_patience, verbose=True)
    res = train_utils._train(train_dataset, test_dataset, criterion, optimizer, fold+1, scheduler=lr_scheduler, 
                            dry_run=True, show_graphs=False, max_epoch=1)
    return 0 if res else -1


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
    res = train_utils._train(train_dataset, test_dataset, criterion, optimizer, fold+1, scheduler=lr_scheduler, 
                            dry_run=True, show_graphs=False, max_epoch=1)
    return 0 if res else -1


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
    res = train_utils._train(train_dataset, test_dataset, criterion, optimizer, fold+1, scheduler=lr_scheduler, 
                            dry_run=True, show_graphs=False, max_epoch=1)
    return 0 if res else -1




def test_run_mobilenetv2_epoch():
    train_utils = TrainingUtilities(model_name="mobilenetv2", parameters_path="test_parameters.yml")
    assert run_mobilenetv2_epoch(train_utils) == 0
    

def test_run_xception_epoch():
    train_utils = TrainingUtilities(model_name="xception", parameters_path="test_parameters.yml")
    assert run_xception_epoch(train_utils) == 0 
    

def test_run_vggm_epoch():
    train_utils = TrainingUtilities(model_name="vggm", parameters_path="test_parameters.yml")
    assert run_vggm_epoch(train_utils) == 0 
   

def test_run_resnext101_32x4d_epoch():
    train_utils = TrainingUtilities(model_name="resnext101_32x4d", parameters_path="test_parameters.yml")
    assert run_resnext101_32x4d_epoch(train_utils) == 0
    

def test_run_inceptionv4_epoch():
    train_utils = TrainingUtilities(model_name="inceptionv4", parameters_path="test_parameters.yml")
    assert run_inceptionv4_epoch(train_utils) == 0
    