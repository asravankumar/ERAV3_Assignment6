import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.network_gap import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = Net()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, which exceeds the limit of 20000"

def test_batch_normalization():
    model = Net()
    has_batchnorm = False
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            has_batchnorm = True
            break
    assert has_batchnorm, "Model does not use Batch Normalization"

def test_dropout():
    model = Net()
    has_dropout = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            has_dropout = True
            break
    assert has_dropout, "Model does not use Dropout"

def test_global_average_pooling():
    model = Net()
    has_gap = False
    for module in model.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            has_gap = True
            break
    assert has_gap, "Model does not use Global Average Pooling" 