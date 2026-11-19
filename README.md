# Polygon-Based Instance Segmentation for Quantitative Severity Estimation of Mango Anthracnose Using YOLOv8-Seg

## 1. Project Overview

This repository contains the Python implementation of a computer vision framework for mango anthracnose detection, instance segmentation and quantitative disease severity estimation using YOLOv8-Seg.

The proposed methodology uses polygon-based instance segmentation to identify mango leaf regions and anthracnose-affected regions. The segmented regions are subsequently used for pixel-wise disease area calculation and quantitative severity estimation.

The study focuses on automated mango leaf disease analysis under real-field orchard conditions, considering variations in illumination, background and image appearance.

## 2. Research Objectives

The main objectives of this research are:

- Detect and segment mango leaf regions.
- Identify anthracnose-affected regions using polygon-based segmentation.
- Validate detected leaf regions before disease analysis.
- Calculate diseased and healthy leaf areas using segmentation masks.
- Estimate disease severity as a percentage of the affected leaf area.
- Classify disease severity into Healthy, Mild, Moderate and Severe categories.

## 3. Proposed Methodology

The proposed research methodology consists of the following stages.

### Stage 1: Field Video Acquisition

Mango leaf videos are collected from orchard environments under real-field conditions. The recorded videos capture variations in illumination, leaf appearance and background conditions.

### Stage 2: Video Frame Extraction

Frames are extracted from the recorded videos using a fixed sampling interval of every 15 frames.

This process generates individual mango leaf images for subsequent dataset preparation.

### Stage 3: Leaf Selection and Quality Screening

The extracted frames are reviewed to identify suitable images for the study.

Poor-quality images, including blurred, duplicate, occluded and unsuitable frames, are excluded during the selection process.

### Stage 4: Manual Polygon Annotation Using Roboflow

The selected images are uploaded to Roboflow for manual polygon annotation.

Two annotation classes are defined:

| Class ID | Class name |
|---|---|
| 0 | Anthracnose |
| 1 | Leaf |

Polygon annotations are used to represent mango leaf boundaries and anthracnose-affected regions.

### Stage 5: Image Preprocessing in Roboflow

The following preprocessing operations are configured in Roboflow:

- Auto-Orient.
- Resize using Stretch to 640 × 640 pixels.

### Stage 6: Dataset Generation and Export

The annotated dataset is generated and exported in YOLOv8 segmentation format.

The dataset is configured using the following partitioning strategy:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The exported dataset contains images, corresponding annotation files and the dataset configuration file (`data.yaml`).

### Stage 7: YOLOv8-Seg Model Training

The exported dataset is used to train the YOLOv8n-Seg instance segmentation model.

The training and validation subsets are used during model development, while the independent test subset is reserved for final performance evaluation.

### Stage 8: Leaf Validation and Disease Segmentation

The trained model is used to identify mango leaf regions and anthracnose-affected regions.

A leaf validation step is incorporated before proceeding with disease analysis.

### Stage 9: Pixel-Wise Disease Area Calculation

The predicted segmentation masks are used to calculate the leaf area and anthracnose-affected area.

Disease severity is estimated using the ratio of diseased leaf pixels to total leaf pixels.

### Stage 10: Quantitative Disease Severity Classification

The estimated disease severity percentage is categorized into predefined severity levels.

## 4. Roboflow Preprocessing and Augmentation

The following image preprocessing settings were configured in Roboflow.

**Preprocessing:**

| Operation | Configuration |
|---|---|
| Auto-Orient | Applied |
| Resize | Stretch to 640 × 640 pixels |

**Augmentation:**

| Augmentation | Configuration |
|---|---|
| Outputs per training example | 3 |
| Horizontal Flip | Applied |
| Rotation | -10° to +10° |
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Noise | Up to 0.1% of pixels |

These augmentation settings were configured for the training dataset.

## 5. Dataset Description

The study uses a customized mango leaf dataset containing 3,000 images extracted from orchard videos.

The images are annotated using polygon-based annotations in Roboflow.

The dataset includes:

- Healthy mango leaf images.
- Anthracnose-affected mango leaf images.
- Leaf segmentation annotations.
- Anthracnose lesion annotations.

The dataset is organized into training, validation and testing subsets using a 70:15:15 partitioning strategy.

The exported dataset is maintained separately from this source-code repository. Dataset access information is provided through the associated research article and dataset repository.

## 6. Model Architecture and Implementation

| Parameter | Description |
|---|---|
| Model architecture | YOLOv8n-Seg |
| Task | Instance segmentation |
| Number of classes | 2 |
| Input image size | 640 × 640 pixels |
| Programming language | Python |
| Deep learning framework | Ultralytics YOLO |
| Image processing library | OpenCV |

## 7. Disease Severity Estimation

Disease severity is calculated as the percentage of anthracnose-affected leaf area relative to the total segmented leaf area.

The severity percentage is expressed as:

**Disease Severity (%) = (Anthracnose-affected leaf area / Total leaf area) × 100**

The severity categories used in the study are:

| Category | Disease severity |
|---|---|
| Healthy | 0% |
| Mild | >0–10% |
| Moderate | >10–25% |
| Severe | >25% |

The classification is based on the calculated disease severity percentage.

## 8. Repository Structure

```text
mango-anthracnose-yolov8-seg/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── raw_videos/
│   ├── extracted_frames/
│   └── roboflow_export/
│
├── src/
│   ├── preprocessing.py
│   ├── extract_frames.py
│   ├── train.py
│   ├── evaluate.py
│   └── severity_estimation.py
│
├── config/
│   └── data.yaml
│
├── notebooks/
├── examples/
├── weights/
├── results/
└── docs/
```

## 9. Installation

The implementation uses Python and the Ultralytics YOLO framework.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 10. Dataset Preparation

Download the corresponding YOLOv8 segmentation dataset export from Roboflow, subject to the dataset's access and licensing conditions.

Extract the dataset into:

```text
dataset/roboflow_export/
```

The exported dataset should contain:

- Training images and annotations.
- Validation images and annotations.
- Testing images and annotations.
- Dataset configuration file (`data.yaml`).

Verify that polygon annotations are preserved for the intended segmentation classes before model training.

## 11. Model Training and Evaluation

The YOLOv8n-Seg model is trained using the exported dataset.

The evaluation stage examines the model's detection and segmentation performance using the evaluation metrics and analysis procedures described in the research article.

Training outputs, model checkpoints, evaluation figures and segmentation visualizations will be organized in the corresponding repository directories.

## 12. Reproducibility

This repository is intended to provide the source code, configuration files and documentation associated with the proposed mango anthracnose detection and severity estimation methodology.

The experimental implementation and dataset configuration are organized to support reproducibility of the reported research.

## 13. Code Availability

The source code associated with this research is available at:

https://github.com/bhargavi23phd7186-glitch/mango-anthracnose-yolov8-seg

A permanent archived version and DOI will be provided through Zenodo after repository archiving.

## 14. Citation

If you use this implementation in your research, please cite the associated research article:

**Polygon-Based Instance Segmentation for Quantitative Severity Estimation of Mango Anthracnose Using YOLOv8-Seg**

The complete publication details will be added after publication.

## 15. License

The source-code license will be specified after confirming the authors' rights to distribute the implementation.

Dataset licensing and attribution requirements are governed separately by the dataset's applicable license.
