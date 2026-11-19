# Extracted Frames

This directory is intended to contain the frames extracted from the original mango orchard videos during dataset preparation.

## Frame Extraction

Frames were extracted from the recorded videos using an OpenCV-based Python script. The extraction process selected frames at regular intervals to reduce redundancy while retaining representative visual information from the recorded videos.

The extraction interval used in the frame extraction workflow was every 15 frames.

## Frame Selection

The extracted frames were reviewed to identify suitable images for further processing and annotation. Frames containing blur, excessive duplication, poor visibility, or unsuitable image quality were excluded during screening.

## Image Format

The extracted frames were saved as image files for subsequent processing and annotation.

## Directory Organization

Frames may be organized into subdirectories corresponding to their source video files to maintain traceability.

Example:

```text
extracted_frames/
├── video_01/
│   ├── video_01_frame_000015.jpg
│   ├── video_01_frame_000030.jpg
│   └── ...
├── video_02/
│   ├── video_02_frame_000015.jpg
│   └── ...
└── README.md
```

## Next Processing Stage

The extracted frames were subjected to quality screening and prepared for annotation. Selected images were subsequently annotated using polygon annotations in Roboflow for mango leaf and anthracnose segmentation.
