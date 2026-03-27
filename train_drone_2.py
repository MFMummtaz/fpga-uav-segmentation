from comet_ml import Experiment
import os
from tqdm import tqdm
from datetime import datetime
import time

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from torchvision import transforms as T

# from data_loader.segmentation_loader import KITTI360_Segmentation_Data
# from data_loader.cityscapes_loader_v2 import CityscapesDataset
# from data_loader.drone_loader import UAVInstanceDataset, get_transforms, collate_fn
from data_loader.drone_loader_antiuav import UAVSegmDataset

# from models.VanillaUNet import VanillaUNet
# from models.unet import UNet
# from models.unet_simple_drone import UnetGenerator
from models.thindyunet import ThinDyUNet
from models.ulite import ULite
from models.custom_light import custom_ULite

from utils.save_model import save_model
from utils.loss_function import get_loss_function
from utils.common import custom_collate_fn, get_optimizer, dice_coeff, seg_miou, pixel_accuracy
# from utils.metrics import SegmentationMetric

def validation(model, criterion, vloader, device='cuda:0'):
    model.eval()
    # total_loss = 0.0
    # total_acc = 0.0
    # total_miou = 0.0
    # total_dice = 0.0

    # Initialize accumulators as tensors on the device to avoid CPU syncs
    total_loss = torch.tensor(0.0, device=device)
    total_acc = torch.tensor(0.0, device=device)
    total_miou = torch.tensor(0.0, device=device)
    total_dice = torch.tensor(0.0, device=device)

    total_batches = len(vloader)
    
    val_process = tqdm(vloader, total=len(vloader))

    with torch.no_grad():
        
        for images, masks in val_process:
            # data preparation
            images = images.to(device)
            masks = masks.to(device)

            # utilize model to predict
            outputs = model(images)
            # Buat BCELogitLoss
            masks = masks.unsqueeze(1).float()
            loss = criterion(outputs, masks)

            # Decode the predictions logits
            pred_masks = (outputs.sigmoid() > 0.9).float()

            # total_loss += loss.item()
            total_loss += loss
            total_acc += pixel_accuracy(pred_masks, masks)
            total_miou += seg_miou(pred_masks, masks)
            total_dice += dice_coeff(pred_masks, masks)
            
            del images
            del masks

        # avg_acc = total_acc / total_batches
        # avg_miou = total_miou / total_batches
        # avg_dice = total_dice / total_batches
        # avg_loss = total_loss / total_batches

        # Move to CPU only once at the very end
        avg_loss = total_loss.item() / total_batches
        avg_acc = total_acc.item() / total_batches
        avg_miou = total_miou.item() / total_batches
        avg_dice = total_dice.item() / total_batches

        print('EVAL METRICS: ')
        print(f"Pixel Accuracy: {avg_acc:.4f} | Mean IoU: {avg_miou:.4f} | Dice Coeff: {avg_dice:.4f}")

    return avg_loss, avg_acc, avg_miou, avg_dice

def train(model, tloader, vloader=None, epochs=None, 
          optimizer=None, starting_epoch=None, last_best_loss=None, 
          device=None):
    
    print("last epoch:", starting_epoch)
    print("last best loss:", last_best_loss)

    if last_best_loss is None:
        best_val_loss = 1500000000.0
    else:
        best_val_loss = last_best_loss

    old_save_filename = None
    
    print("START TRAINING PROCESS")

    for epoch in range(starting_epoch, epochs + 1):
        print('This is %d-th epoch' % epoch)
        model.train()
        epoch_start_time = time.time()
        loss_epoch = 0.

        train_process = tqdm(tloader, total=len(tloader))

        for images, masks in train_process:
            # data preparation
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # utilize model to predict
            outputs = model(images)

            # calculate loss and run backpropagation
            # Buat BCELogitLoss
            # masks = masks.unsqueeze(1).float()
            loss = criterion(outputs, masks)
            loss.backward()

            optimizer.step()
            loss_epoch += loss.item()

            # Upload Each Batch Loss to Comet
            experiment.log_metric("training batch loss", loss.item())

            del images
            del masks

        epoch_end_time = time.time()
        print(f"epoch time: {epoch_end_time - epoch_start_time}")
        loss_epoch /= len(tloader)

        # Upload Epoch Training Loss to Comet
        experiment.log_metric("training epoch loss", loss_epoch)

        # Print training epoch loss
        print("***")
        print(f"Total Loss for Epoch {epoch}: {loss_epoch}")
        print()

        print("START VALIDATION PROCESS")
        # enter validation process function
        val_loss_epoch, acc, mIoU, dice = validation(model, criterion, vloader=vloader, device=device)

        # Upload Epoch Validation Loss to Comet
        experiment.log_metric("validation epoch loss", val_loss_epoch)
        experiment.log_metric("validation MIoU", mIoU)
        experiment.log_metric("validation Accuracy", acc)

        # Print validation epoch loss
        print()
        print("***")
        print(f"Validation Loss for Epoch {epoch}: {val_loss_epoch}")
        print(f"Validation MIoU for Epoch {epoch}: {mIoU}")
        print(f"Validation Acc for Epoch {epoch}: {acc}")
        print()

        # scheduler
        scheduler.step(val_loss_epoch)

        # Upload Learning Rate After Optimizer update to Comet
        experiment.log_metric("epoch", epoch)
        experiment.log_metric("learning rate", optimizer.param_groups[0]["lr"])
        print("current learning rate:", str(optimizer.param_groups[0]["lr"]))
        
        # save checkpoint with the best validation score
        if val_loss_epoch < best_val_loss:
            print("saving model...")
            print("best val loss achieved")
            savefilename = f'{ckpt_dir}/{config.model.name}-e{epoch}-best-val.pth.tar'
            save_model(
                model, optimizer, 
                scheduler, epoch, 
                val_loss_epoch, 
                savefilename
            )
            best_val_loss = val_loss_epoch

            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

if __name__ == "__main__":

    ############### ENV INITIALIZATION ##################
    torch.set_printoptions(precision=10)
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method("spawn")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # Config setup
    # config = OmegaConf.load('./models/config/config_ulite_custom.yaml')
    config = OmegaConf.load('./models/config/config_thindyunet.yaml')
    dataset_cfg = config.dataset
    trainer_cfg = config.trainer
    model_cfg = config.model
    ckpt_dir = config.trainer.checkpoint.save_dir

    # Get data loaders
    transform = T.Compose([
                    # T.Resize(dataset_cfg.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.31328324, 0.32151696, 0.31460182], 
                                std=[0.23343998, 0.24014007, 0.23295579])
                ])

    # Set device for this process
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_savepath = os.path.join('./weights')

    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)

    MODEL_DATE  = datetime.now().strftime("%m:%d_%H:%M")
    print(f"\n{'='*60}")

    COMET_API = "ZbHnIRQBveKFZWoUx3BpIds0x"
    experiment = Experiment(
        api_key=COMET_API,
        project_name="segmentation",
        workspace="mummtaz18",
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=False,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
    )

    experiment.set_name(f"{model_cfg.name}_{MODEL_DATE}")

    dataset_train = UAVSegmDataset(dataset_cfg.root, 2, transform, "train")
    dataset_val = UAVSegmDataset(dataset_cfg.root, 2, transform, "val")
    
    # Create dataloaders
    train_loader = DataLoader(dataset_train, batch_size=dataset_cfg.batch_size,
                              num_workers=dataset_cfg.num_workers, shuffle=True,
                              pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=dataset_cfg.batch_size,
                            shuffle=True, num_workers=dataset_cfg.num_workers,
                            pin_memory=True, collate_fn=custom_collate_fn)
    
    print('Train dataset size:', len(dataset_train))
    print('Train batch dataset size:', len(train_loader))

    print('Val dataset size:', len(dataset_val))
    print('Val batch dataset size:', len(val_loader))
    
    ############### TRAINING INITIALIZATION ##################

    # Define the model
    model = ThinDyUNet(in_channels=model_cfg.in_channels, 
                       start_out_channels=model_cfg.start_out_channels, 
                       num_class=model_cfg.num_classes, 
                       size=model_cfg.num_blocks, 
                       padding=model_cfg.num_padding, 
                       upsample=model_cfg.is_upsample).to(device)
    
    # model = custom_ULite().to(device)

    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Model total parameters: {pytorch_total_params:,}')
    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Optimizer & Loss Function & Lr Scheduler
    optimizer = get_optimizer(trainer_cfg.optimizer, model, trainer_cfg.lr)
    criterion = get_loss_function(trainer_cfg.loss_fn)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=trainer_cfg.scheduler.factor,
                                                           patience=trainer_cfg.scheduler.patience,
                                                           cooldown=trainer_cfg.scheduler.cooldown)

    if trainer_cfg.pretrained_path:
        print(f"Loading weights from {trainer_cfg.pretrained_path}")
        checkpoint = torch.load(trainer_cfg.pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        last_epoch = checkpoint["epoch"]
        last_val_loss = checkpoint["val_loss"]
        print(f"Loaded pre-trained model weights from {trainer_cfg.pretrained_path}")
    else:
        last_epoch, last_val_loss = 0, None

    ############### TRAINING START ##################
    train(
        model=model,
        tloader=train_loader,
        vloader=val_loader,
        epochs=trainer_cfg.epochs,
        optimizer=optimizer,
        starting_epoch=last_epoch,
        last_best_loss=last_val_loss,
        device=device,
    )
    
    print("TRAIN FINISH")
    print("################")