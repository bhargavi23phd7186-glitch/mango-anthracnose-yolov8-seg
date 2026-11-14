# mango-anthracnose-yolov8-seg
Python implementation of mango anthracnose detection and disease severity estimation using YOLOv8-Seg instance segmentation.
# Mango Anthracnose Detection and Severity Estimation Using YOLOv8-Seg

## Project Overview

This repository contains the Python implementation of a computer vision pipeline for mango anthracnose detection and disease severity estimation using YOLOv8-Seg instance segmentation.

The proposed approach uses polygon-based instance segmentation to identify mango leaf regions and anthracnose-affected areas, followed by pixel-wise disease area calculation and quantitative severity estimation.

The implementation is intended to support automated mango leaf disease analysis under real-field image conditions.

## Key Features

- Mango leaf detection and instance segmentation.
- Leaf validation before disease analysis.
- Anthracnose lesion segmentation.
- Pixel-wise calculation of diseased and healthy leaf areas.
- Quantitative disease severity estimation.
- Severity classification into Healthy, Mild, Moderate and Severe categories.
- Model evaluation using detection and segmentation performance metrics.

## Methodology

The overall processing pipeline consists of:

1. Input image acquisition.
2. Image preprocessing and normalization.
3. YOLOv8-Seg-based leaf detection and segmentation.
4. Leaf validation.
5. Anthracnose disease segmentation.
6. Pixel-wise disease area calculation.
7. Disease severity percentage estimation.
8. Disease severity classification.

## Severity Classification

Disease severity is calculated as the percentage of diseased leaf area relative to the total leaf area.

The classification thresholds used in the study are:

| Category | Disease severity |
|---|---|
| Healthy | 0% |
| Mild | >0–10% |
| Moderate | >10–25% |
| Severe | >25% |

## Dataset

The study uses a collection of 3,000 mango leaf images extracted from orchard videos and annotated using polygon-based segmentation.

The dataset includes healthy and anthracnose-affected mango leaf images.

Dataset access and annotation information will be provided through the Data Availability statement of the associated research article.

## Model and Implementation

- Model: YOLOv8n-Seg
- Framework: Ultralytics YOLO
- Programming language: Python
- Image processing: OpenCV
- Input resolution: 640 × 640 pixels

## Repository Structure

- `notebooks/` – Jupyter notebooks for experiments and analysis.
- `src/` – Python source code for the proposed pipeline.
- `config/` – Configuration files.
- `weights/` – Model weight information.
- `results/` – Evaluation results.
- `examples/` – Example inputs and outputs.

## Installation

The required Python packages and installation instructions will be provided in `requirements.txt`.

## Reproducibility

The source code is being organized to support reproducibility of the mango anthracnose detection and severity estimation experiments described in the associated research article.

## Citation

If you use this implementation in your research, please cite the associated research article.

**Article title:** Polygon-Based Instance Segmentation for Quantitative Severity Estimation of Mango Anthracnose Using YOLOv8-Seg

The permanent archived source-code version and DOI will be added after Zenodo publication.

## Code Availability

The source code for this study is available through this GitHub repository. A DOI-linked archived version will be provided through Zenodo.

## License

A suitable open-source license will be specified after confirming the authors' rights to distribute the implementation.
