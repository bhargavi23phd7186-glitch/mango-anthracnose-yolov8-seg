
import cv2
import os
import glob

# ==========================================
# VIDEO FRAME EXTRACTION
# ==========================================

# Input video folder
video_folder = "dataset/raw_videos"

# Output folder for extracted frames
output_folder = "dataset/extracted_frames"

# Extract every 15th frame
frame_interval = 15

# Create output directory
os.makedirs(output_folder, exist_ok=True)

# Supported video formats
video_extensions = (".mp4", ".avi", ".mov", ".mkv")

# Find video files
video_files = [
    f for f in glob.glob(os.path.join(video_folder, "*"))
    if f.lower().endswith(video_extensions)
]

if not video_files:
    raise FileNotFoundError(
        f"No video files found in {video_folder}"
    )

total_saved_frames = 0

# ==========================================
# PROCESS EACH VIDEO
# ==========================================

for video_path in video_files:

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    print(f"\nProcessing video: {video_name}")

    # Create separate folder for each video
    video_output = os.path.join(
        output_folder, video_name
    )
    os.makedirs(video_output, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        continue

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS:", fps)
    print("Frame resolution:", width, "x", height)
    print("Total frames:", total_frames)

    frame_count = 0
    saved_count = 0

    # ======================================
    # EXTRACT FRAMES
    # ======================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:

            frame_filename = os.path.join(
                video_output,
                f"{video_name}_frame_{frame_count:06d}.jpg"
            )

            success = cv2.imwrite(
                frame_filename, frame
            )

            if success:
                saved_count += 1

        frame_count += 1

    cap.release()

    total_saved_frames += saved_count

    print("Frames extracted:", saved_count)
    print("Output folder:", video_output)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n================================")
print("VIDEO FRAME EXTRACTION COMPLETED")
print("================================")
print("Total videos:", len(video_files))
print("Total extracted frames:", total_saved_frames)
print("Frame interval:", frame_interval)
print("Output directory:", output_folder)
