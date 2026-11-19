
import cv2
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIR = ROOT / "dataset" / "raw_videos"
FRAME_DIR = ROOT / "dataset" / "extracted_frames"

# Extract every 15th frame
FRAME_INTERVAL = 15

# Supported video formats
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


# ==========================================
# VALIDATION
# ==========================================

if type(FRAME_INTERVAL) is not int or FRAME_INTERVAL <= 0:
    raise ValueError(
        "FRAME_INTERVAL must be a positive integer."
    )

if not VIDEO_DIR.exists():
    raise FileNotFoundError(
        f"Raw video directory does not exist: {VIDEO_DIR}"
    )

if not VIDEO_DIR.is_dir():
    raise NotADirectoryError(
        f"Expected a directory: {VIDEO_DIR}"
    )


# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

FRAME_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FIND VIDEO FILES
# ==========================================

video_files = sorted([
    file for file in VIDEO_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in VIDEO_EXTENSIONS
])

if not video_files:
    raise FileNotFoundError(
        f"No supported video files found in: {VIDEO_DIR}"
    )


# ==========================================
# PROCESS EACH VIDEO
# ==========================================

total_saved_frames = 0
total_processed_videos = 0
total_failed_videos = 0
total_failed_saves = 0

for video_path in video_files:

    video_name = video_path.stem

    print("\n===================================")
    print("Processing video:", video_name)
    print("===================================")

    # Create separate folder for each video
    video_output = FRAME_DIR / video_name
    video_output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("ERROR: Could not open video:", video_path)

        total_failed_videos += 1
        cap.release()
        continue

    # ======================================
    # VIDEO METADATA
    # ======================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames_metadata = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        total_frames_metadata / fps
        if fps > 0
        else 0
    )

    print("FPS:", fps)
    print("Resolution:", width, "x", height)
    print("Frame count (metadata):", total_frames_metadata)
    print("Duration (seconds):", round(duration, 2))

    # ======================================
    # FRAME EXTRACTION
    # ======================================

    frame_count = 0
    saved_count = 0
    failed_saves = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Extract every FRAME_INTERVAL-th frame
        if frame_count % FRAME_INTERVAL == 0:

            frame_filename = (
                f"{video_name}_frame_{frame_count:06d}.png"
            )

            frame_path = video_output / frame_filename

            # Save frame in lossless PNG format
            success = cv2.imwrite(
                str(frame_path),
                frame
            )

            if success:
                saved_count += 1

            else:
                failed_saves += 1
                print("ERROR saving:", frame_path)

        frame_count += 1

    cap.release()

    # ======================================
    # VIDEO SUMMARY
    # ======================================

    print("\nVideo processing summary:")
    print("Frames successfully read:", frame_count)
    print("Frames extracted:", saved_count)
    print("Failed image saves:", failed_saves)
    print("Output folder:", video_output)

    total_saved_frames += saved_count
    total_failed_saves += failed_saves
    total_processed_videos += 1


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")
print("VIDEO FRAME EXTRACTION COMPLETED")
print("===================================")

print("Total videos found:", len(video_files))
print("Videos processed successfully:", total_processed_videos)
print("Videos failed to open:", total_failed_videos)
print("Total extracted frames:", total_saved_frames)
print("Total failed image saves:", total_failed_saves)
print("Frame extraction interval:", FRAME_INTERVAL)
print("Output directory:", FRAME_DIR)
