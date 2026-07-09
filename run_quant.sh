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


#!/bin/bash

#echo "Creating environment..."
#conda env create -f code/configs/environment.yaml
#source activate torch_seg

echo "Preparing dataset..."
#VAL_IMG=/workspace/dataset/Panoptic_dataset_tarot/Instance_updated/UAV-Image/val/
#VAL_MASK=/workspace/dataset/Panoptic_dataset_tarot/Instance_updated/UAV-Mask/val/
DATASET=/workspace/dataset/cityscapes/
WEIGHTS=checkpoint


# echo "Conducting testing and miou evaluation with: "
# echo " SemanticFPN(ResNet18) with input_size: 256x512. "

export PYTHONPATH=${PWD}:${PYTHONPATH} 
export W_QUANT=1

GPU_ID=1
#echo "====> perform SemanticFPN(ResNet18) with input_size = 256x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn_res18/final_best.pth.tar  --quant_mode calib --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn_res18/final_best.pth.tar  --quant_mode test --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn_res18/final_best.pth.tar  --quant_mode test --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune

#echo "====> perform SemanticFPN(Non-Pretrained) with input_size = 256x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn/checkpoint.pth.tar  --quant_mode calib --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn/checkpoint.pth.tar  --quant_mode test --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset citys --model fpn --backbone resnet18 --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/fpn/checkpoint.pth.tar  --quant_mode test --quant_dir quantize_result_fpn_res18_256_512 --fast_finetune

#cho "====> perform Unet with input_size = 256x512..."
#CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model unet --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/unet/unet_try_quant.pth.tar  --quant_mode calib --quant_dir quantize_result_unet_256_512 --fast_finetune
#CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset citys --model unet --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/unet/unet_try_quant.pth.tar  --quant_mode test --quant_dir quantize_result_unet_256_512 --fast_finetune
#CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset citys --model unet --crop-size 256 --data-folder ${DATASET} --weight ${WEIGHTS}/unet/unet_try_quant.pth.tar  --quant_mode test --quant_dir quantize_result_unet_256_512 --fast_finetune

# echo "====> perform Unet Drone with input_size = 544x960..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model unet --crop-size 256 --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/unet/model_best_linknet.pth.tar  --quant_mode calib --quant_dir quantize_result_linknet_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model unet --crop-size 256 --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/unet/model_best_linknet.pth.tar  --quant_mode test --quant_dir quantize_result_linknet_256_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset drone --model unet --crop-size 256 --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/unet/model_best_linknet.pth.tar --quant_mode test --quant_dir quantize_result_linknet_256_512 --fast_finetune

# echo "====> perform fpn Drone with input_size = 512x512..."
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model fpn --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/fpn/coba_fpncustomnet6_best.pth.tar  --quant_mode calib --quant_dir quantize_result_fpn_custom6_512_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model fpn --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/fpn/coba_fpncustomnet6_best.pth.tar  --quant_mode test --quant_dir quantize_result_fpn_custom6_512_512 --fast_finetune
# CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset drone --model fpn --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/fpn/coba_fpncustomnet6_best.pth.tar --quant_mode test --quant_dir quantize_result_fpn_custom6_512_512 --fast_finetune

echo "====> perform custom net Drone with input_size = 512x512..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model custom --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/custom/custom5_cont_good.pth.tar  --quant_mode calib --quant_dir quantize_result_customnet5_cont_512_512 --fast_finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model custom --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/custom/custom5_cont_good.pth.tar   --quant_mode test --quant_dir quantize_result_customnet5_cont_512_512 --fast_finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset drone --model custom --data-folder ${DATASET} --num-classes 2 --weight ${WEIGHTS}/drone/custom/custom5_cont_good.pth.tar  --quant_mode test --quant_dir quantize_result_customnet5_cont_512_512 --fast_finetune

echo "Test finishes!"
