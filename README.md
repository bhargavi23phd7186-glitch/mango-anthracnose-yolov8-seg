# Mango Anthracnose Detection and Disease Severity Estimation Using YOLOv8-Seg

## 1. Project Overview

This repository contains the Python implementation of a computer vision framework for mango anthracnose detection, instance segmentation and quantitative disease severity estimation using YOLOv8-Seg.

The proposed approach uses polygon-based annotations to represent mango leaf regions and anthracnose-affected areas. The segmentation masks are used for pixel-wise disease area calculation and quantitative severity estimation.

The implementation is intended to support automated mango leaf disease analysis under real-field image conditions.

## 2. Key Features

- Mango leaf detection and instance segmentation.
- Leaf validation before disease analysis.
- Anthracnose lesion segmentation.
- Polygon-based representation of leaf and disease regions.
- Pixel-wise calculation of diseased and healthy leaf areas.
- Quantitative disease severity estimation.
- Severity classification into Healthy, Mild, Moderate and Severe categories.
- Model evaluation using detection and segmentation performance metrics.

## 3. Research Methodology

The proposed processing pipeline consists of the following stages:

1. Mango orchard image and video acquisition.
2. Video frame extraction, where applicable.
3. Image annotation using Roboflow.
4. Image preprocessing and augmentation.
5. YOLOv8-Seg dataset export.
6. Dataset partitioning.
7. YOLOv8n-Seg model training.
8. Leaf detection and validation.
9. Anthracnose lesion segmentation.
10. Pixel-wise disease area calculation.
11. Disease severity percentage estimation.
12. Disease severity classification.

## 4. Dataset Description

The study uses a collection of 3,000 mango leaf images obtained from orchard data and video-derived frames.

Image annotation was performed using Roboflow. The dataset contains two classes:

| Class ID | Class Name |
|---|---|
| 0 | Anthracnose |
| 1 | Leaf |

The dataset was configured for the following partitioning:

| Dataset | Percentage | Number of Images |
|---|---:|---:|
| Training | 70% | 2,100 |
| Validation | 15% | 450 |
| Testing | 15% | 450 |
| **Total** | **100%** | **3,000** |

These counts describe the original dataset partition. Augmented training image counts may differ.

### Dataset Availability

The dataset was prepared using Roboflow and is maintained separately from this source-code repository.

The dataset export and annotation information are documented in the `dataset/` directory.

## 5. Image Preprocessing

The following preprocessing operations were configured in Roboflow:

- Auto-Orient.
- Resize using Stretch to 640 × 640 pixels.

### Data Augmentation

The Roboflow augmentation configuration included:

| Augmentation | Configuration |
|---|---|
| Outputs per training example | 3 |
| Flip | Horizontal |
| Rotation | -10° to +10° |
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Noise | Up to 0.1% of pixels |

These augmentation settings were configured for the training dataset.

## 6. Model and Implementation

| Parameter | Description |
|---|---|
| Model | YOLOv8n-Seg |
| Framework | Ultralytics YOLO |
| Task | Instance segmentation |
| Programming language | Python |
| Image processing | OpenCV |
| Input resolution | 640 × 640 pixels |
| Number of classes | 2 |

## 7. Disease Severity Estimation

Disease severity is calculated as the percentage of diseased leaf area relative to the total leaf area.

The severity calculation is expressed as:

\[
S_r = \frac{A_d}{A_l}\times 100
\]

where:

- \(S_r\) = Disease severity percentage.
- \(A_d\) = Pixel area of the anthracnose-affected region.
- \(A_l\) = Total segmented leaf area.

### Severity Classification

The severity thresholds used in the study are:

| Category | Disease Severity |
|---|---|
| Healthy | 0% |
| Mild | Greater than 0% to 10% |
| Moderate | Greater than 10% to 25% |
| Severe | Greater than 25% |

## 8. Repository Structure

```text
mango-anthracnose-yolov8-seg/
│
├── dataset/
│   ├── raw_videos/
│   ├── extracted_frames/
│   ├── roboflow_export/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   └── README.md
│
├── config/
│   └── README.md
│
├── notebooks/
│
├── examples/
│   └── README.md
│
├── weights/
│   └── README.md
│
├── results/
│   └── README.md
│
├── requirements.txt
└── README.md
```

Additional training, evaluation and severity estimation scripts will be documented as their implementations are finalized.

## 9. Installation

Python 3.11 or a compatible Python environment is recommended.

Install the required packages using:

```bash
pip install ultralytics opencv-python numpy
```

Additional dependencies will be listed in `requirements.txt`.

## 10. Dataset Preparation

Download the corresponding YOLOv8 segmentation dataset export from Roboflow, subject to the dataset's access and licensing conditions.

Extract the dataset into:

```text
dataset/roboflow_export/
```

The exported dataset is expected to contain:

- Training images and annotations.
- Validation images and annotations.
- Testing images and annotations.
- Dataset configuration file (`data.yaml`).

The annotation files should be checked to ensure that polygon segmentation annotations are preserved for the intended segmentation classes.

## 11. Experimental Results

The `results/` directory is intended to contain the experimental outputs, including:

- Model training curves.
- Confusion matrix.
- Segmentation evaluation results.
- ROC curve.
- Disease severity analysis.
- Example segmentation predictions.

Only verified experimental results will be included in the repository.

## 12. Reproducibility

This repository is intended to provide the Python implementation, configuration files and documentation required to understand and reproduce the mango anthracnose detection and severity estimation methodology.

The final training configuration, dataset version, evaluation procedure and experimental settings will be documented alongside the corresponding implementation.

## 13. Code Availability

The source code for this study is available at:

https://github.com/bhargavi23phd7186-glitch/mango-anthracnose-yolov8-seg

A permanent archived version and DOI will be added following repository archiving through Zenodo.

## 14. Dataset Attribution

The Roboflow dataset export identifies its license as CC BY 4.0.

Appropriate attribution and applicable licensing conditions must be followed when accessing or redistributing the dataset.

## 15. Citation

If you use this implementation in your research, please cite the associated research article:

**Polygon-Based Instance Segmentation for Quantitative Severity Estimation of Mango Anthracnose Using YOLOv8-Seg**

The complete publication citation will be added after publication.

## 16. License

The source-code license will be specified after confirming the authors' rights to distribute the implementation.
