# Source Code

This directory contains the Python source code for the mango anthracnose detection and severity estimation project using YOLOv8-Seg.

The source code is organized around the main stages of the proposed image-processing and segmentation workflow.

## 1. Processing Pipeline

The project workflow consists of the following stages:

1. Video frame extraction
2. Image selection and preprocessing
3. Mango leaf localization and validation
4. Anthracnose lesion segmentation
5. Pixel-based disease area calculation
6. Disease severity estimation
7. Severity category assignment

## 2. Source Files

The following files are intended to contain the implementation of the project:

| File | Purpose |
|---|---|
| `extract_frames.py` | Extracts frames from orchard videos at a specified interval. |
| `preprocessing.py` | Performs image preprocessing operations. |
| `train.py` | Configures and trains the YOLOv8-Seg model. |
| `evaluate.py` | Evaluates the trained model using the test dataset. |
| `severity_estimation.py` | Calculates disease severity from leaf and lesion masks. |
| `test_frame_extraction.py` | Checks the frame extraction output. |

Only source files that have been implemented and verified should be included in the final repository.

## 3. Image Preprocessing

The preprocessing implementation may include:

- Image resizing
- Noise reduction using median filtering
- Contrast enhancement using CLAHE
- Color-space processing

The exact preprocessing steps used for the final experiment should be consistent with the methodology reported in the manuscript.

## 4. Model Training

The training stage uses the YOLOv8 segmentation architecture with two classes:

- Class 0: anthracnose
- Class 1: leaf

The dataset configuration file specifies the training, validation, and testing image directories.

## 5. Disease Severity Estimation

Disease severity is estimated using the segmented leaf and anthracnose lesion regions.

The calculation is based on the ratio of the diseased leaf area to the total leaf area:

\[
\text{Disease Severity (\%)} =
\frac{\text{Anthracnose Lesion Area}}
{\text{Total Leaf Area}} \times 100
\]

The severity classification thresholds should match the final implementation and the manuscript.

## 6. Running the Source Code

Run Python scripts from the repository root directory so that relative dataset paths resolve correctly.

For example:

```bash
python src/extract_frames.py
```

The commands for training, evaluation, and severity estimation will be documented after their corresponding scripts are finalized.

## 7. Reproducibility

For reproducible results:

- Use the documented Python and package versions.
- Verify dataset paths before execution.
- Use the same dataset split and class mapping.
- Preserve the training and evaluation settings used in the reported experiment.
- Keep generated results separate from source code.

