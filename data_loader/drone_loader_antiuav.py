import os
from glob import glob
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

class UAVSegmDataset(Dataset):
    def __init__(self, root, n_classes, transform, mode):
        self.root_input = os.path.join(root, "input")
        self.root_mask = os.path.join(root, "labels")
        self.transforms = transform
        self.n_classes = n_classes
        self.mode = mode
        self.img_type = ['infrared', 'visible']
        self.images, self.masks = self.__get_file_path(self.root_input, self.root_mask, self.mode)
        # print(f"Number of files: {len(self.images)} {len(self.masks)}")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)

        if self.transforms:
            image = self.transforms(image)

        mask = np.array(mask)
        mask[mask > 0] = 1 # Collapse 255 down to 1
        mask = torch.from_numpy(mask).float()

        return image, mask

    def __get_file_path(self, root, root_mask, mode):
        image_list, mask_list = [], []
        i = 0
        
        if mode == 'train':
            root = os.path.join(root, 'train')
            root_mask = os.path.join(root_mask, 'train')

        elif mode == 'val':
            root = os.path.join(root, 'val')
            root_mask = os.path.join(root_mask, 'val')

        elif mode == 'test':
            root = os.path.join(root, 'show_paper')
            root_mask = os.path.join(root_mask, 'show_paper')

        for seq in sorted(os.listdir(root)):

            for img_type in self.img_type:
                image_path = sorted(glob(os.path.join(root, seq, img_type, "*.jpg")))
                mask_path = sorted(glob(os.path.join(root_mask, seq, img_type, "*.png")))

                image_list += image_path
                mask_list += mask_path
            
            # If want test for only 1 seq of data
            # break
            
        # print(f"Number of files: {len(image_list)} {len(mask_list)}")
        # print('Image list:', image_list[-1])
        # print('Mask list:', mask_list[-1])
        if mode == 'train':
            combined = list(zip(image_list, mask_list))
            random.seed(42)
            random.shuffle(combined)

            # Take 50%
            combined = combined[:len(combined)//2]
            image_list, mask_list = zip(*combined)

        # For semantic dataset
        mask_lookup = set(['/'.join(x.split('/')[-3:])[:-9] for x in mask_list])
        image_list = [x for x in image_list if '/'.join(x.split('/')[-3:])[:-4] in mask_lookup]
        
        # For panoptic dataset
        # mask_lookup = {os.path.basename(x).replace('_mask', '').split('.')[0] for x in mask_list}
        # image_list = [x for x in image_list if os.path.basename(x).split('.')[0] in mask_lookup]
        
        print(f"Number of files: {len(image_list)} {len(mask_list)}") 
        
        assert len(image_list) == len(mask_list), 'Mismatch total images and masks in the dataset'
        return image_list, mask_list
    
    def get_stats(self, image_list):
        pixel_sum = np.zeros(3)
        pixel_sq_sum = np.zeros(3)
        count = 0

        for img_path in image_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
            
            pixel_sum += np.mean(img, axis=(0, 1))
            pixel_sq_sum += np.mean(img**2, axis=(0, 1))
            count += 1

        mean = pixel_sum / count
        # std = sqrt(E[X^2] - (E[X])^2)
        std = np.sqrt((pixel_sq_sum / count) - (mean**2))
    
        return mean, std
    
if __name__ == '__main__':
    # DATASET_PATH = '/home/wicomai/datasets/Panoptic_dataset/semantic'
    DATASET_PATH = '/home/wicomai/datasets/UAVSegmentationDataset'
    NUM_CLASSES = 2

    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    dataset = UAVSegmDataset(DATASET_PATH, NUM_CLASSES, transform, "train")
    print('Dataset size:', len(dataset))

    mean, std = dataset.get_stats(dataset.images)
    print(f'Mean: {mean}, Std: {std}')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print('Dataloader size:', len(dataloader))
    
    i=0
    for qq, (images, masks) in enumerate(dataloader):
        i+=1
        print(i)
        print(images.shape, masks.shape)
        break