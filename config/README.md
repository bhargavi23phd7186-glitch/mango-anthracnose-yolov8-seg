# Configuration Files

This directory contains configuration documentation and settings used in the mango anthracnose detection and severity estimation project.

## 1. Dataset Configuration

The dataset was annotated and exported using Roboflow in YOLOv8 format.

**Roboflow project details:**
- Workspace: `vidhu-9wo34`
- Project: `mango-pylut`
- Dataset version: `2`
- Annotation format: YOLOv8
- License: CC BY 4.0

Dataset link:  
https://universe.roboflow.com/vidhu-9wo34/mango-pylut/dataset/2

## 2. Class Configuration

The project uses two classes:

| Class ID | Class name |
|---|---|
| 0 | anthracnose |
| 1 | leaf |

The class order must remain consistent between the dataset YAML file, annotation labels, model training, and evaluation.

## 3. Dataset Split

The intended dataset partition is:

| Dataset split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The reported partition for the 3,000-image dataset is 2,100 training images, 450 validation images, and 450 testing images. Confirm the actual exported folder counts before training.

## 4. Image Preprocessing

The Roboflow dataset version uses the following preprocessing settings:

- Auto-Orient
- Resize: Stretch to 640 × 640 pixels

## 5. Data Augmentation

The Roboflow training dataset configuration includes:

- Horizontal Flip
- Rotation: −10° to +10°
- Brightness: −15% to +15%
- Exposure: −10% to +10%
- Noise: up to 0.1% of pixels

These settings describe the Roboflow dataset configuration.

## 6. YOLOv8 Dataset YAML

The dataset YAML file should be placed in the dataset export directory, alongside the `train`, `valid`, and `test` folders.

Example configuration:

```yaml
path: .
train: train/images
val: valid/images
test: test/images

nc: 2
names: ['anthracnose', 'leaf']
```

The relative paths must match the actual exported dataset folder structure.

## 7. Model Configuration

The project uses the YOLOv8 segmentation architecture for leaf and anthracnose instance segmentation.

Training parameters, including image size, number of epochs, batch size, and device, should be documented in the training script based on the final experiment settings.

## 8. Important Notes

- Keep the class IDs and class names consistent.
- Verify that both classes have the intended polygon segmentation annotations before training.
- Check the dataset YAML paths before running the training script.
- Do not upload private API keys or access tokens.
- Keep the original dataset export unchanged when making configuration adjustments.
