# FPGA UAV Semantic Segmentation

Real-time semantic segmentation of aerial imagery onboard Unmanned Aerial Vehicles (UAVs) accelerated on FPGA embedded platforms.

---

## 📌 Project Overview

This repository provides an end-to-end framework for training, optimizing, and deploying lightweight deep learning models (e.g., U-Net, ENet) for **real-time UAV aerial image segmentation** onto KR260 FPGA Development Board. 

By leveraging hardware acceleration by the board, this project achieves low-latency and energy-efficient inference suitable for edge deployment on autonomous drones.

---

## 🧰 Prerequisites & Hardware Setup

### Hardware
- **Target FPGA Board:** AMD Xilinx Zynq UltraScale+ KR260 Board
- **Host Workstation:** x86_64 PC with Linux (Ubuntu 20.04 / 22.04 LTS recommended) and NVIDIA GPU (for model training/quantization)