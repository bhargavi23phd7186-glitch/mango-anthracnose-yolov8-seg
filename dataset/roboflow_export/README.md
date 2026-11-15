
# Roboflow Dataset

This directory is intended to organize the dataset exported from Roboflow for mango anthracnose instance segmentation.

## 1. Polygon Annotation

Manual polygon annotations were created for:
- Mango leaf regions
- Anthracnose lesion regions

## 2. Image Preprocessing

The following preprocessing operations were applied in Roboflow:

- Auto-Orient
- Resize: Stretch to 640 × 640 pixels

## 3. Data Augmentation

The Roboflow augmentation configuration included:

- Outputs per training example: 3
- Horizontal flip
- Rotation: -10° to +10°
- Brightness: -15% to +15%
- Exposure: -10% to +10%
- Noise: Up to 0.1% of pixels

Augmentation settings were configured for the training data.

## 4. Dataset Export

The dataset was exported in YOLOv8-Seg segmentation format.

Expected organization:

- Training images and labels
- Validation images and labels
- Testing images and labels
- data.yaml configuration file

## 5. Dataset Availability

The exported dataset is maintained separately and is not included in this repository.
