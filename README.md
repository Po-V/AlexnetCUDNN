# AlexNet Inference Using CUDA

This repository provides a CUDA-based implementation of AlexNet for performing inference on input images using the CUDNN library. It loads pre-trained weights from ONNX models (not included in this repository) and is based on NVIDIA’s CUDA sample code for LeNet, adapted for AlexNet architecture.

## Overview
This project implements the AlexNet model for image classification, optimized for GPU-based inference using CUDA and cuDNN libraries. The implementation includes custom layers, fully connected and convolutional layers.

**Link to pre-trained model weights: **: [ONNX AlexNet Model Weights](https://github.com/onnx/models/tree/main/validated/vision/classification/alexnet)

## Requirements
To use this code, ensure the following dependencies are installed:
- **CUDA Toolkit** (Version 10.1 or newer)
- **cuDNN Library** (for accelerated GPU processing)
- **OpenCV** (for image processing)

This project has been tested with CUDA Compute Capability `sm_75` but should work on any compatible device with minor adjustments.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Po-V/AlexnetCUDNN.git
   cd AlexnetCUDNN

2. **Set up weights**:
Download the ONNX pre-trained weights for AlexNet in binary format amd store them in model_weights folder.

3. **Compile and run the project**:
    ```bash
    make
    ./alexnet

## TODO
- Add support for fp16