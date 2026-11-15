
# Model Configuration

## Dataset
- Source: Roboflow
- Annotation: Polygon-based instance segmentation
- Format: YOLOv8-Seg

## Image preprocessing
- Auto-Orient
- Stretch resize to 640 × 640 pixels

## Dataset split
- Training: 70%
- Validation: 15%
- Testing: 15%

## Augmentation
- Outputs per training example: 3
- Horizontal flip
- Rotation: -10° to +10°
- Brightness: -15% to +15%
- Exposure: -10% to +10%
- Noise: Up to 0.1% of pixels

## Model
- Architecture: YOLOv8n-Seg
- Input size: 640 × 640
- Training epochs: 150
- Batch size: 8
