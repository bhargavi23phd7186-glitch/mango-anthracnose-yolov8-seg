# Source Code

This directory contains the source code used for dataset preparation and the mango anthracnose detection and severity estimation workflow.

## Code Organization

The source code is organized to support the processing of video-derived images and preparation of the dataset for YOLOv8-Seg training.

## Preprocessing

The preprocessing code is intended to support image preparation operations such as:

- Image resizing
- Image quality improvement
- Image preprocessing for model input

The preprocessing implementation is provided in `preprocessing.py`.

## Dataset Preparation

The dataset preparation workflow includes:

1. Extracting frames from orchard videos.
2. Reviewing extracted frames for image quality.
3. Preparing images for annotation.
4. Exporting the annotated dataset from Roboflow in YOLOv8 format.

## Model Training and Evaluation

The proposed approach uses YOLOv8n-Seg for mango leaf and anthracnose instance segmentation.

The model training and evaluation scripts will be organized in this directory as they are added to the repository.

## Reproducibility

The source code is provided to support reproducibility of the dataset preparation and model development workflow.

Please refer to the main repository README for the overall methodology and experimental workflow.
