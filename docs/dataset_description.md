# Dataset Description

## Dataset Overview

The mango anthracnose dataset was developed for leaf detection, disease segmentation, and disease severity estimation.

The dataset contains 3,000 mango leaf images obtained from field video recordings.

The images represent real-world acquisition conditions, including variations in illumination, background, and image appearance.

## Classes

The dataset contains two annotation classes:

| Class ID | Class Name |
|---|---|
| 0 | leaf |
| 1 | anthracnose |

## Dataset Split

The dataset is divided into training, validation, and testing subsets.

| Dataset Split | Number of Images | Percentage |
|---|---:|---:|
| Training | 2,100 | 70% |
| Validation | 450 | 15% |
| Testing | 450 | 15% |
| Total | 3,000 | 100% |

## Annotation Format

The images are annotated using polygon-based segmentation annotations.

The annotations are stored in YOLO segmentation format, with normalized polygon coordinates.

Each annotation represents an object belonging to either the leaf or anthracnose class.

## Dataset Structure

The expected dataset structure is:

```text
mango_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/


## Dataset Access

Dataset availability and download instructions will be provided through the associated dataset repository or archival repository.

Please refer to the project README for the latest dataset access information.
