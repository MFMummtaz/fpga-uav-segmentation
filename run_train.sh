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

DATASET=/workspace/dataset/cityscapes/
# DATASET=/workspace/dataset/cityscapes/
WEIGHTS=checkpoint/drone

# DATASET=/workspace/dataset/mapillary/
# WEIGHTS=checkpoint/mapil

GPU_ID=0

echo "Conducting training with: "
# echo " SemanticFPN(ResNet18) with input_size: 256x512. "
# echo " SemanticUnet with input_size: 256x512. "

export PYTHONPATH=${PWD}:${PYTHONPATH}

# echo "====> perform SemanticFPN(ResNet18) with input_size = 256x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train/train.py --lr 1e-4 --dataset citys --model fpn --backbone resnet18 --crop-size 512 --data-folder ${DATASET} --batch-size 32 --weight ${WEIGHTS}/fpn/model_best_pretrained_new.pth.tar

# echo "====> perform SemanticUnet with input_size = 256x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train/train.py --lr 1e-3 --dataset citys --model unet --data-folder ${DATASET} --batch-size 18 --workers 10 --start-features 64 
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/train/train.py --lr 1e-5 --dataset citys --model unet --data-folder ${DATASET} --batch-size 18 --workers 10 --start-features 64 --weight ${WEIGHTS}/unet/model_best_unet64_new.pth.tar


# echo "====> perform Semanticfpn with input_size = 256x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model fpn --num-classes 2 --data-folder ${DATASET} --batch-size 32 --workers 10  

# echo "====> perform unet drone with input_size = 544x960..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model unet --num-classes 2 --data-folder ${DATASET} --batch-size 24 --workers 10  
# CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model unet --num-classes 2 --data-folder ${DATASET} --batch-size 24 --workers 10  

# echo "====> perform fpn custom drone with input_size = 512x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model fpn --num-classes 2 --data-folder ${DATASET} --batch-size 24 --workers 10  

echo "====> perform custom drone with input_size = 512x512..." 
CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model custom --num-classes 2 --data-folder ${DATASET} --batch-size 24 --workers 10  --weight ${WEIGHTS}/custom/model_best_customnet5.pth.tar

# echo "====> perform thindyunet drone with input_size = 512x512..." 
# CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python code/train/train_drone.py --lr 1e-3 --dataset drone --model thindyunet --num-classes 2 --data-folder ${DATASET} --batch-size 24 --workers 10 --weight ${WEIGHTS}/thindyunet/checkpoint_thindyunet.pth.tar