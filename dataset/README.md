# Dataset

This directory contains the dataset-related resources used in the mango anthracnose detection and severity estimation study.

## Dataset Workflow

The dataset was prepared from mango orchard videos and processed through the following stages:

1. **Video acquisition:** Videos were recorded under field conditions in a mango orchard.
2. **Frame extraction:** Frames were extracted from the recorded videos at regular intervals.
3. **Frame screening:** Extracted frames were reviewed to remove unsuitable images, including blurred, duplicate, or poor-quality frames.
4. **Annotation:** Selected images were annotated using polygon annotations in Roboflow to identify mango leaf regions and anthracnose regions.
5. **Preprocessing:** Images were auto-oriented and resized to 640 × 640 pixels using the Roboflow workflow.
6. **Dataset export:** The annotated dataset was exported in YOLOv8 format for model training and evaluation.

## Directory Structure

```text
dataset/
├── README.md
├── raw_videos/
│   └── README.md
├── extracted_frames/
│   └── README.md
└── roboflow_export/
    └── README.md
```

## Dataset Organization

- `raw_videos/`: Documentation and information related to the original orchard videos.
- `extracted_frames/`: Documentation related to frames extracted from the source videos.
- `roboflow_export/`: Documentation for the annotated dataset exported from Roboflow.

## Dataset Annotation

The dataset contains two annotation classes:

| Class ID | Class Name |
|---|---|
| 0 | anthracnose |
| 1 | leaf |

Polygon annotations are used to represent leaf regions and anthracnose lesions for instance segmentation.

## Dataset Partitioning

The dataset was divided into training, validation, and testing subsets using a 70:15:15 split.

The dataset configuration file (`data.yaml`) specifies the corresponding image directories and class names.

## Dataset Availability

The dataset files are maintained separately from this source-code repository. Dataset access and associated annotation information are described in the `roboflow_export/` directory.

Please refer to the corresponding directory README files for additional information about each dataset preparation stage.
