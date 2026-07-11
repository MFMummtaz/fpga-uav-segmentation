# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# MIT License

# Copyright (c) 2019 Hengshuang Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import torch.nn.functional as F

import code.utils as utils
from code.utils.misc import save_checkpoint
from code.utils.metrics import batch_pix_accuracy, pixel_accuracy, batch_intersection_union
from code.utils.metrics import *
from code.datasets.drone_loader_antiuav import UAVSegmDataset
from code.configs.model_config import Options

from code.models.ulite import ULite
from code.models.fpn_custom import FPN
from code.models.linknet import LinkNet
from code.models.thindyunet import ThinDyUNet
from code.models.custom_model import Novelty_ULite

import logging
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
#torch.backends.cudnn.benchmark = True

class Combined2ClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # outputs: [B, 2, H, W], targets: [B, H, W]
        ce_loss = self.ce(outputs, targets)
        
        # Simple Multiclass Dice logic
        probs = F.softmax(outputs, dim=1)
        # Convert target to one-hot for dice math
        targets_one_hot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        
        inter = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        dice_loss = 1 - ((2. * inter + 1e-6) / (union + 1e-6)).mean()
        
        return ce_loss + dice_loss


def collate_fn(batch):
    # 1. Separate images and targets into standard Python lists
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    
    # 2. Stack the list of images into a single, solid 4D tensor!
    # (This is exactly what conv2d is begging for)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    # 3. Leave targets as a list of dictionaries
    return images, masks

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None, None
    
    data, masks = zip(*batch)
    return torch.stack(data), torch.stack(masks)

class Trainer():
    def __init__(self, args):
        self.args = args

        # data transforms cityscapes
        # input_transform = transform.Compose([
        #     transform.ToTensor(),
        #     # transform.Normalize([.485, .456, .406], [.229, .224, .225])
        #     ])

        # data transforms drones
        input_transform = transform.Compose([transform.ToTensor(), 
                                             transform.Normalize(mean=[0.31328324, 0.32151696, 0.31460182], 
                                                                 std=[0.23343998, 0.24014007, 0.23295579])
                                            ])

        # dataset [CHANGE] !!!
        root_dir="/workspace/dataset/UAVSegmentationDataset/"
        
        trainset = UAVSegmDataset(root_dir, 2, input_transform, "train")
        testset = UAVSegmDataset(root_dir, 2, input_transform, "val")
    
        self.trainloader = data.DataLoader(
            trainset, batch_size=18, shuffle=True,
            num_workers=10, collate_fn=custom_collate_fn, drop_last=False, pin_memory=False
        )

        self.valloader = data.DataLoader(
            testset, batch_size=18, shuffle=False,
            num_workers=10, collate_fn=custom_collate_fn, drop_last=False, pin_memory=True
        )

        self.nclass = args.num_classes
        self.best_pred = 0.0 

        # model
        if args.model == "unet":
            # model = ULite()
            model = LinkNet(classes=2)
        elif args.model == "fpn":
            model = FPN(num_blocks=[2, 4, 8, 4], num_classes=2)
        elif args.model == "custom":
            model = Novelty_ULite(num_classes=2)
        elif args.model == "thindyunet":
            model = ThinDyUNet(in_channels=3, 
                               start_out_channels=64, 
                               num_class=2, 
                               size=7, 
                               padding=1, 
                               upsample=True)
        else:
            print("undefined models")
            exit()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'[INFO] Model total parameters: {pytorch_total_params:,}')

        # optimizer using different LR
        params_list = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.weight_decay)

        # criterions
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = Combined2ClassLoss()

        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            # self.criterion = self.criterion.cuda()
        # resuming checkpoint
        if args.weight is not None:
            if not os.path.isfile(args.weight):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cuda:0')
            checkpoint['state_dict'] = OrderedDict([(k[5:], v) if 'base' in k else (k, v) for k, v in checkpoint['state_dict'].items()])
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})" \
                  .format(args.weight, checkpoint['epoch']))

        # set up scheduler for decreased the LR by a factor of 0.1 
        # if the validation results didn't improve over 5 consecutive checks.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=10, eps=1e-6)                                               

    def training(self, epoch):
        train_loss = 0.0
        loss_epoch = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)
            image = image.cuda()
            target =  target.cuda()
            outputs = self.model(image)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            target = target.long()
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # Upload Each Batch Loss to Comet
            # experiment.log_metric("training batch loss", loss.item())

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        loss_epoch /= len(self.trainloader)
        # Upload Epoch Training Loss to Comet
        # experiment.log_metric("training epoch loss", loss_epoch)


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            if isinstance(outputs, tuple):# for aux
                outputs = outputs[0]
            loss = self.criterion(outputs, target)
            correct, labeled = batch_pix_accuracy(outputs.data, target)
            inter, union = batch_intersection_union(outputs.data, target, self.nclass)
            
            return correct, labeled, inter, union, loss

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        # val_loss_epoch = 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            # if torch_ver == "0.3":
            #     image = Variable(image, volatile=True).cuda()
            #     correct, labeled, inter, union, loss = eval_batch(self.model, image, target)
            # else:
            with torch.no_grad():
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True).long()
                correct, labeled, inter, union, loss = eval_batch(self.model, image, target)

                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
                total_loss += loss.item()
            
            if i % 10 == 0:
                tbar.set_description(f'Batch {i}/{len(self.valloader)}')
    
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
            
        # experiment.log_metric("validation mIoU", mIoU)
        total_loss /= len(self.valloader)

        # Upload Epoch Validation Loss to Comet
        # experiment.log_metric("validation epoch loss", total_loss)
        
        self.scheduler.step(total_loss)
        
        print("current Val MIoU:", mIoU)
        print("current Val pixAcc:", pixAcc)
        print("current epoch:", epoch)
        print("current learning rate:", str(self.optimizer.param_groups[0]["lr"]))
        print("\n")

        # Upload Learning Rate After Optimizer update to Comet
        # experiment.log_metric("epoch", epoch)
        # experiment.log_metric("learning rate", self.optimizer.param_groups[0]["lr"])

        new_pred = mIoU
        if new_pred > self.best_pred:
            print("saving model...")
            print("best val loss achieved")
            is_best = True
            self.best_pred = new_pred
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, self.args, is_best)

        


if __name__ == "__main__":
    # MODEL_DATE  = datetime.now().strftime("%m:%d_%H:%M")
    args = Options().parse()
    for key, val in args._get_kwargs():
        logging.info(key+' : '+str(val))
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
        
