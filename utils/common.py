import torch
import numpy as np
import os

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None, None
    
    data, masks = zip(*batch)
    return torch.stack(data), torch.stack(masks)

def get_optimizer(optimizer, model, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    else:
        raise ValueError(f'Optimizer {optimizer} not supported')
    
# def pixel_accuracy(pred_mask, true_mask):
#     correct = (pred_mask == true_mask).sum().item()
#     total = true_mask.numel()
#     return correct / total

def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    total = true_mask.numel()
    return correct / total

# def seg_miou(pred_mask, true_mask):
#     pred_mask = pred_mask.bool()
#     true_mask = true_mask.bool()
#     intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
#     union = (pred_mask | true_mask).float().sum((1, 2, 3))
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return iou.mean().item()

def seg_miou(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    
    dims = tuple(range(1, pred_mask.ndimension()))
    intersection = (pred_mask & true_mask).sum(dims).float()
    union = (pred_mask | true_mask).sum(dims).float()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# def dice_coeff(pred_mask, true_mask):
#     pred_mask = pred_mask.bool()
#     true_mask = true_mask.bool()
#     intersection = (pred_mask & true_mask).float().sum((1, 2, 3))
#     dice = (2. * intersection + 1e-6) / (pred_mask.float().sum((1, 2, 3)) + true_mask.float().sum((1, 2, 3)) + 1e-6)
#     return dice.mean()

def dice_coeff(pred_mask, true_mask):
    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()
    
    dims = tuple(range(1, pred_mask.ndimension()))
    intersection = (pred_mask & true_mask).sum(dims).float()
    
    denom = pred_mask.sum(dims).float() + true_mask.sum(dims).float()
    dice = (2. * intersection + 1e-6) / (denom + 1e-6)
    return dice.mean()