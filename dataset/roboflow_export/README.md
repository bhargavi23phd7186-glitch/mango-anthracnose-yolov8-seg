# Roboflow Dataset Export

## 1. Dataset Overview

This directory is intended to document and organize the customized mango anthracnose dataset exported from Roboflow.

The dataset was prepared for mango leaf detection, anthracnose lesion segmentation and quantitative disease severity estimation using YOLOv8-Seg.

## 2. Dataset Source

- Platform: Roboflow
- Project: mango-pylut
- Workspace: vidhu-9wo34
- Original export version: 2
- Annotation format selected: YOLOv8
- Dataset license indicated in export metadata: CC BY 4.0

Roboflow project:

https://universe.roboflow.com/vidhu-9wo34/mango-pylut/dataset/2

## 3. Annotation Classes

The dataset configuration defines two classes:

| Class ID | Class name |
|---|---|
| 0 | Anthracnose |
| 1 | Leaf |

Polygon annotations are used in the Roboflow annotation workflow to represent mango leaf regions and anthracnose-affected regions.

The exported annotation files must be verified to ensure that polygon coordinates are preserved for both classes before YOLOv8-Seg training.

## 4. Image Preprocessing

The following preprocessing operations were configured in Roboflow:

| Operation | Configuration |
|---|---|
| Auto-Orient | Applied |
| Resize | Stretch to 640 × 640 pixels |

## 5. Data Augmentation

The following augmentation settings were configured in Roboflow:

| Augmentation | Configuration |
|---|---|
| Outputs per training example | 3 |
| Horizontal Flip | Applied |
| Rotation | -10° to +10° |
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Noise | Up to 0.1% of pixels |

These settings were configured for the training dataset.

## 6. Dataset Partitioning

The Roboflow dataset was configured using the following split:

| Subset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The original dataset split and the number of exported augmented images should be considered separately.

## 7. Exported Dataset Structure

The expected YOLOv8 dataset organization is:

```text
roboflow_export/
│
├── data.yaml
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

The images and corresponding annotation files are organized according to their respective dataset subsets.

## 8. Dataset Configuration

The `data.yaml` file specifies:

- Training image directory.
- Validation image directory.
- Testing image directory.
- Number of classes.
- Class names.

The dataset configuration defines two classes: anthracnose and leaf.

## 9. Dataset Availability

The exported dataset is maintained separately from this GitHub source-code repository.

Dataset access and redistribution are subject to the applicable Roboflow dataset license and attribution requirements.

The dataset files are not included in this directory.

## 10. Usage

Download the corresponding YOLOv8 segmentation dataset export from Roboflow and extract it into this directory.

Verify the image folders, annotation files, polygon coordinates and `data.yaml` paths before model training.
