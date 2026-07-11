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

echo "Preparing dataset..."
WEIGHTS=checkpoint/drone/custom/custom5_cont_good.pth.tar
QUANTI_DIR=quantize_result_customnet5_cont_512_512 


export PYTHONPATH=${PWD}:${PYTHONPATH} 
export W_QUANT=1

GPU_ID=0

echo "====> Perform Quantization of Lightweight Custom Net Drone with input_size = 512x512..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model custom --num-classes 2 --weight ${WEIGHTS}  --quant_mode calib --quant_dir ${QUANTI_DIR} --fast_finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --eval --dataset drone --model custom --num-classes 2 --weight ${WEIGHTS}   --quant_mode test --quant_dir ${QUANTI_DIR} --fast_finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python code/test/test.py --dump_xmodel --eval --dataset drone --model custom --num-classes 2 --weight ${WEIGHTS}  --quant_mode test --quant_dir ${QUANTI_DIR} --fast_finetune

echo "Test finishes!"
