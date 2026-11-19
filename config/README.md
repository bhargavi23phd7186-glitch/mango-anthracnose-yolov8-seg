# Configuration Files

This folder contains configuration information required to prepare and run the mango anthracnose instance segmentation experiments.

## Model Configuration

The proposed system uses the YOLOv8n-Seg model for instance segmentation of mango leaves and anthracnose lesions.

| Parameter | Configuration |
|---|---|
| Model | YOLOv8n-Seg |
| Task | Instance segmentation |
| Number of classes | 2 |
| Class 0 | anthracnose |
| Class 1 | leaf |
| Input image size | 640 × 640 pixels |
| Dataset format | YOLOv8 segmentation format |

## Dataset Configuration

The dataset configuration file is named `data.yaml`. It specifies the paths to the training, validation, and testing image directories, along with the class names.

The dataset configuration file is obtained from the Roboflow YOLOv8 export.

## Dataset Split

The intended dataset partition is:

- Training: 70%
- Validation: 15%
- Testing: 15%

## Training Configuration

The model training configuration should record the actual values used in the final experiment, including:

- Number of training epochs
- Image size
- Batch size
- Model weights
- Dataset configuration path
- Training output directory

The exact training settings should be taken from the final training script or training log.

## Reproducibility

Configuration files and training settings are provided to help reproduce the experiments. Any changes to the dataset version, model settings, or training parameters should be documented.
