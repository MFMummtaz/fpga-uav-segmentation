import torch
import os

def save_model(model, optimizer, lr_scheduler, epoch, loss, path):
    
    base_dir = os.path.dirname(path)

    if (not os.path.exists(base_dir)):
        os.makedirs(base_dir)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss,
    }, path)