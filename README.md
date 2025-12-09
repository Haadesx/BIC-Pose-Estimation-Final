# Event-Based Arm Pose Estimation

This project implements an end-to-end pipeline for arm pose estimation using event camera data. It includes tools for pseudo-label generation (using MediaPipe on source videos) and a PointNet-based neural network for regression of joint coordinates from event point clouds.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running the Code](#running-the-code)
  - [1. Generate Pseudo-Labels](#1-generate-pseudo-labels)
  - [2. Model Training](#2-model-training)
  - [3. Inference & Visualization](#3-inference--visualization)
- [Project Structure](#project-structure)

## Prerequisites
- **OS:** Windows / Linux / macOS (Windows 10/11 tested)
- **Python:** 3.8+ (Recommended 3.10)
- **GPU:** NVIDIA GPU with CUDA support is highly recommended for training.
  - Tested on: NVIDIA GeForce RTX 3060

## Installation

1. **Clone/Unzip** this project folder to your local machine.
2. **Create a virtual environment** (Anaconda or venv):
   ```bash
   conda create -n eventpose python=3.10
   conda activate eventpose
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `mediapipe` and `protobuf`, ensure `tensorflow` is NOT installed in the same environment, as it often conflicts.*

## Data Setup

This project uses the **DHP19** and **MoCA** datasets. Due to size constraints, these are not included in the submission.

1. **Directories**: Ensure you have a data directory (e.g., `Y:\BIC\DHP19EPC_dataset\custom`).
2. **Event Data**: Place your event `.npy` files (e.g., `frame_000000_cam2.npy`) in subfolders within the custom dataset directory.
3. **Source Videos**: For label generation, you need the original `dvs-video.avi` files from the MoCA dataset.

**Expected Structure:**
```
DHP19EPC_dataset/
  └── custom/
      ├── CARROT_TR_0/
      │   ├── frame_000000_cam2.npy
      │   ├── frame_000001_cam2.npy
      │   └── ...
      ├── TRANSPORTING_TR_2/
      │   └── ...
      └── dvs-video/
          └── (Symlinks or copies of .avi files if using the generation script recursively)
```

## Running the Code

### 1. Generate Pseudo-Labels
If you do not have ground truth labels, generating them from the source videos is required.
```bash
python generate_labels_mediapipe.py
```
- **Configuration**: Edit the `dataset_dir` path in `generate_labels_mediapipe.py` to point to your MoCA videos.
- **Output**: This will create `*_label.npy` files next to your event files.

### 2. Model Training
Train the PointNet model on the event data.
```bash
python train_arm_pose.py --epochs 50 --batch_size 8
```
- **Key Arguments**:
  - `--epochs`: Number of training passes (Default: 5). Recommended: 50+.
  - `--batch_size`: Adjust based on GPU VRAM (Default: 8).
  - `--learning_rate`: Default 0.001.
- **Outputs**:
  - Checkpoints saved to `checkpoints/`.
  - `loss_curve.png` generated at the end of training.

### 3. Inference & Visualization
Visualizes the model's predictions against the generated labels.
```bash
python visualize_inference.py
```
- **Input**: Uses `best_model.pth` (included in repo) or your own `checkpoints/`.
  - *Note: Update the script to point to `best_model.pth` if not using local training checkpoints.*
- **Output**: Saves `inference_result.png` showing:
  - **Green**: Event Data (Rasterized)
  - **Yellow**: Ground Truth (Pseudo-Labels)
  - **Red**: Model Predictions

## Model Architecture

The core model is a modified **PointNet** architecture adapted for pose regression.

### 1. Encoder (Feature Extraction)
The network takes raw event point clouds `(N, 5)` as input (x, y, t, p, mask).
- **Layers:** A series of 1D Convolutions with Batch Norm and LeakyReLU.
  - `Conv1d`: 5 -> 64
  - `Conv1d`: 64 -> 64
  - `Conv1d`: 64 -> 128
  - `Conv1d`: 128 -> 256
- **Feature Aggregation:** Intermediate features from these layers are concatenated.
- **Global Feature:** A final `Conv1d` (512 -> 1024) allows for global context, followed by **Adaptive Max & Average Pooling** to aggregate point features into a global vector.

### 2. Decoder (Pose Regression)
The global feature vector is fed into a series of fully connected layers:
- **MLP Block:** Linear(2048 -> 1024) -> Linear(1024 -> 128 * Num_Joints).
- **Heads:** Separate Linear layers project features to X and Y coordinate heatmaps (SimDR representation).
  - `mlp_head_x`: 128 -> Sensor_Width (346)
  - `mlp_head_y`: 128 -> Sensor_Height (260)

## Project Structure
- `dataset/`: Contains `DHP19EPC.py` (Data Loader) and helper classes.
- `models/`: Contains `PointNet.py` and other architecture definitions.
- `train_arm_pose.py`: Main training script.
- `generate_labels_mediapipe.py`: Label generation utility.
- `visualize_inference.py`: Evaluation script.
