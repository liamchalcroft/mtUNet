# Multi-Task nnU-Net for Joint Segmentation and Classification

This repository contains the official implementation of [Paper Title] (TBA), which extends the nnU-Net framework to perform joint segmentation and classification tasks.

## System Requirements

This is a direct fork from nnUNet, and so it is expected that the following requirements from nnUNet should hold for mtUNet.

### Operating System
- Linux (Ubuntu 18.04, 20.04, 22.04)
- Windows 10/11
- macOS 13+

### Hardware Requirements for Training
- **GPU**: NVIDIA GPU with at least 10GB VRAM (RTX 3080/3090, RTX 4080/4090, A5000)
- **CPU**: Minimum 6 cores (12 threads), recommended 8+ cores
- **RAM**: 64GB recommended
- **Storage**: SSD (M.2 PCIe Gen 3 or better)

Example workstation configuration:
- CPU: AMD Ryzen 5900X/5950X or Intel i9-13900K
- GPU: NVIDIA RTX 4090 (24GB)
- RAM: 64GB DDR4/DDR5
- Storage: NVMe SSD

### Hardware Requirements for Inference
- GPU with 4GB+ VRAM (recommended)
- CPU-only inference is supported but slower
- 32GB RAM minimum

### Software Requirements
- Python 3.9 or newer
- CUDA 11.7+ (for GPU support)
- PyTorch 2.0+

## Installation

We strongly recommend installing in a virtual environment:

```bash
# Create and activate conda environment
conda create -n mtnn python=3.9
conda activate mtnn

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Clone repository
git clone https://github.com/liamchalcroft/mtUNet.git
cd mtUNet

# Install requirements
pip install -e .

# Set environment variables for data paths
export nnUNet_raw="/path/to/raw/data"
export nnUNet_preprocessed="/path/to/preprocessed/data"
export nnUNet_results="/path/to/results"
```

### Data Augmentation Workers
For optimal training performance, set the number of data augmentation workers according to your CPU/GPU ratio:
```bash
# Recommended values:
# RTX 3090: export nnUNet_n_proc_DA=12
# RTX 4090: export nnUNet_n_proc_DA=16
# A5000: export nnUNet_n_proc_DA=16
```

This value may need adjustment based on your specific hardware configuration and the number of input modalities and classes in your dataset.

## Dataset

Our implementation is tested on the Pancreas-CT dataset. The dataset should be organized in the following structure:

```
Dataset001_Pancreas/
├── imagesTr/
│   ├── case_0.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_0.nii.gz
│   └── ...
├── imagesTs/
│   ├── case_0.nii.gz
│   └── ...
└── dataset.json
```

The `dataset.json` should contain task-specific information including class labels for the classification task.

## Preprocessing

The nnU-Net framework handles preprocessing automatically. To preprocess your data:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -pl nnUNetPlannerResEncM
```

This command will:
- Analyze dataset properties
- Create preprocessing plans
- Generate preprocessed data for training
- Follow the SOTA 'Residual Encoder (M) U-Net' architecture

## Training

To train the multi-task model:

```bash
# Train a single fold
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans

# Train all folds
nnUNetv2_train 1 3d_fullres all -p nnUNetResEncUNetMPlans

# Validate on validation set from training
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetMPlans --val
```

Key hyperparameters:
- Initial learning rate: 1e-2
- Batch size: 2
- Optimizer: SGD with momentum (0.99)
- Training epochs: 1000
- Multi-task loss weights: Automatically determined

## Trained Models

Pre-trained models are available here:
- [Multi-task nnU-Net Model](https://drive.google.com/file/d/1--ZTR60PZ4nJf1LOG16hQAFL9_Yvu9zX/view?usp=share_link) trained on Pancreas-CT dataset

## Inference

To run inference on new cases:

```bash
# Predict single cases
nnUNetv2_predict_multitask -i INPUT_FOLDER -o OUTPUT_FOLDER -d 1 -p nnUNetResEncUNetMPlans -tr nnUNetMultiTaskTrainer -c 3d_fullres -f 0

# Predict with ensemble
nnUNetv2_predict_multitask -i INPUT_FOLDER -o OUTPUT_FOLDER -d 1 -p nnUNetResEncUNetMPlans -tr nnUNetMultiTaskTrainer -c 3d_fullres -f all
```

## Evaluation

To evaluate model performance:

```bash
# Generate evaluation metrics and plots
nnUNetv2_evaluate_folder -ref REFERENCE_FOLDER -pred PREDICTION_FOLDER

# Generate plots only
nnUNetv2_plot_results METRICS_JSON_FILE OUTPUT_FOLDER
```

## Results

Our method achieves the following performance on the Pancreas-CT dataset:

### Segmentation Results
| Structure | Dice Score | Surface Dice (2mm) | Hausdorff 95 |
|-----------|------------|-------------------|---------------|
| Pancreas  | 90.9%     | 93.0%            | 3.1mm        |
| Lesion    | 77.5%     | 72.2%            | 5.1mm        |

### Classification Results
| Metric    | Subtype 0 | Subtype 1 | Subtype 2 | Overall |
|-----------|-----------|-----------|-----------|---------|
| Accuracy  | 94.4%     | 69.4%     | 75.0%     | 69.4%   |
| Precision | 100.0%    | 57.7%     | 100.0%    | 82.4%   |
| Recall    | 77.8%     | 100.0%    | 25.0%     | 69.4%   |

## Contributing

We welcome contributions to improve the codebase. Please feel free to submit issues and pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- We thank the nnU-Net team for their excellent framework
- We thank the contributors of the Pancreas-CT dataset
