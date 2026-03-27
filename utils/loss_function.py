import torch
import segmentation_models_pytorch as smp

def get_loss_function(loss_fn):
    if loss_fn == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == 'BCEWithLogitsLoss':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_fn == 'DiceLoss':
        return smp.losses.DiceLoss(mode='binary')
    elif loss_fn == 'IoULoss':
        return smp.losses.JaccardLoss(mode='binary')
    else:
        raise ValueError(f'Loss function {loss_fn} not supported')

