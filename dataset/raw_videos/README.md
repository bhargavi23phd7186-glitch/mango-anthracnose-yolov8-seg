# Raw Videos

This directory is intended to document the original mango orchard videos used as the source material for dataset preparation.

## Video Acquisition

Videos were recorded under field conditions in a mango orchard to capture mango leaves exhibiting healthy and anthracnose-affected conditions.

The recorded videos provide the source material for extracting images used in the dataset preparation workflow.

## Video Processing

The raw videos were processed using an OpenCV-based frame extraction workflow.

Frames were extracted at regular intervals of every 15 frames to reduce redundancy and obtain representative images from the recorded videos.

## Data Organization

Original video files are maintained separately from the processed image dataset.

The raw video files are not included in this GitHub repository to avoid unnecessarily large repository size and to preserve the original data separately.

## Subsequent Processing

The extracted frames were subjected to image quality screening. Selected frames were subsequently annotated using polygon annotations in Roboflow to identify mango leaf regions and anthracnose lesions.

## Data Availability

The original video data and associated dataset information may be made available by the corresponding author upon reasonable request, subject to applicable data-sharing conditions.
