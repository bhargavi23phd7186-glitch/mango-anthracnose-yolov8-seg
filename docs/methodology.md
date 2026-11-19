# Methodology

## Overview

This project presents a computer vision pipeline for mango anthracnose detection, instance segmentation, and disease severity estimation using YOLOv8-Seg.

The workflow consists of video frame extraction, image screening and preprocessing, polygon annotation, model training, model evaluation, disease severity estimation, and visualization.

## Video Frame Extraction

Frames are extracted from orchard videos using a predefined frame interval. The extracted frames are saved for subsequent image screening and dataset preparation.

## Image Screening and Preprocessing

Extracted frames are screened for suitability. Selected images are resized to the required model input resolution of 640 × 640 pixels.

The preprocessing workflow includes image resizing and image enhancement operations implemented in the preprocessing scripts.

## Dataset Annotation

Mango leaves and anthracnose-affected regions are annotated using polygon masks.

The dataset contains two classes:

- Leaf
- Anthracnose

Annotations are stored in YOLO segmentation format.

## Model Training

YOLOv8n-Seg is used for instance segmentation.

The model is trained using the prepared dataset configuration with an input image size of 640 × 640 pixels.

## Model Evaluation

The trained model is evaluated using the independent test dataset.

Detection and segmentation performance are assessed using precision, recall, F1-score, and mean Average Precision (mAP).

## Disease Severity Estimation

Disease severity is estimated using the proportion of anthracnose-affected pixels relative to leaf pixels.

The severity percentage is calculated as:

Disease Severity (%) = (Anthracnose Area / Leaf Area) × 100

## Severity Classification

The estimated severity percentage is categorized as:

| Category | Severity |
|---|---|
| Healthy | 0% |
| Mild | >0–10% |
| Moderate | >10–25% |
| Severe | >25% |

## Image-Level Classification

Images are classified as healthy or diseased based on the presence of anthracnose detection.

Image-level evaluation includes confusion matrix analysis, classification metrics, and ROC-AUC analysis.
