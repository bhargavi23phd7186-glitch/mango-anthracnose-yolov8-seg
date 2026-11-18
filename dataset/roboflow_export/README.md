# Roboflow Dataset Export

## 1. Dataset Overview

The mango anthracnose dataset used in this research was prepared and annotated using Roboflow.

The dataset contains mango leaf images with two annotation classes:

| Class ID | Class Name |
|---|---|
| 0 | Anthracnose |
| 1 | Leaf |

## 2. Polygon Annotation

Polygon annotations were used to represent mango leaf regions and anthracnose-affected regions.

The annotations are intended for instance segmentation using YOLOv8-Seg.

## 3. Image Preprocessing

The following preprocessing operations were configured in Roboflow:

- Auto-Orient
- Resize: Stretch to 640 × 640 pixels

## 4. Data Augmentation

The Roboflow augmentation configuration included:

| Augmentation | Configuration |
|---|---|
| Outputs per training example | 3 |
| Flip | Horizontal |
| Rotation | -10° to +10° |
| Brightness | -15% to +15% |
| Exposure | -10% to +10% |
| Noise | Up to 0.1% of pixels |

These settings were configured for the training dataset.

## 5. Dataset Partitioning

The dataset was configured using the following split:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

## 6. Export Format

The dataset was downloaded from Roboflow in YOLOv8 format for the instance segmentation workflow.

The expected directory structure is:

```text
roboflow_export/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

## 7. Dataset Configuration

The Roboflow dataset configuration identifies:

- Workspace: vidhu-9wo34
- Project: mango-pylut
- Export version: 2
- Number of classes: 2
- Class names: anthracnose and leaf

## 8. Dataset Access

The exported dataset is maintained separately from this GitHub repository.

Dataset access and redistribution are subject to the applicable Roboflow dataset license and attribution requirements.

The exported annotation files should be verified to ensure polygon coordinates are preserved for both classes before YOLOv8-Seg training.
