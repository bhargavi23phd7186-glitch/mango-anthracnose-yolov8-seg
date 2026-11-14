
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "dataset" / "extracted_frames"
OUTPUT_DIR = ROOT / "dataset" / "processed_frames"

IMAGE_SIZE = 640

SUPPORTED_FORMATS = (
    ".jpg", ".jpeg", ".png", ".bmp"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# RESIZE WHILE PRESERVING ASPECT RATIO
# ==========================================

def resize_with_padding(image, size=640):

    height, width = image.shape[:2]

    scale = min(size / width, size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Create 640 x 640 padded image
    canvas = np.ones(
        (size, size, 3),
        dtype=np.uint8
    ) * 114

    top = (size - new_height) // 2
    left = (size - new_width) // 2

    canvas[
        top:top + new_height,
        left:left + new_width
    ] = resized

    return canvas.astype(np.uint8)


# ==========================================
# CHECK INPUT DIRECTORY
# ==========================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Input directory not found: {INPUT_DIR}"
    )

# ==========================================
# FIND EXTRACTED FRAMES
# ==========================================

image_files = sorted([
    file for file in INPUT_DIR.rglob("*")
    if file.suffix.lower() in SUPPORTED_FORMATS
])

if not image_files:
    raise FileNotFoundError(
        f"No extracted images found in {INPUT_DIR}"
    )

accepted_count = 0
rejected_count = 0
processed_count = 0

print("Total extracted frames:", len(image_files))

# ==========================================
# MANUAL QUALITY SCREENING
# ==========================================

for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Unreadable image:", image_path)
        rejected_count += 1
        continue

    print("\nReviewing:", image_path.name)

    cv2.imshow("Frame Quality Screening", image)

    print("Press Y = Accept frame")
    print("Press N = Reject frame")
    print("Press Q = Quit")

    key = cv2.waitKey(0) & 0xFF

    if key in (ord("q"), ord("Q")):
        print("Screening stopped by user.")
        break

    elif key in (ord("y"), ord("Y")):

        processed_image = resize_with_padding(
            image, IMAGE_SIZE
        )

        # Preserve video-specific folder structure
        relative_path = image_path.relative_to(INPUT_DIR)

        output_path = (
            OUTPUT_DIR /
            relative_path.parent /
            f"{image_path.stem}.png"
        )

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        # Save as PNG
        success = cv2.imwrite(
            str(output_path),
            processed_image
        )

        if success:
            accepted_count += 1
            processed_count += 1
            print("Accepted and saved:", output_path)

        else:
            print("Could not save:", output_path)
            rejected_count += 1

    elif key in (ord("n"), ord("N")):

        rejected_count += 1
        print("Rejected:", image_path.name)

    else:

        print("Invalid key. Frame skipped.")
        rejected_count += 1

cv2.destroyAllWindows()

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")
print("FRAME PREPROCESSING COMPLETED")
print("===================================")

print("Total extracted frames:", len(image_files))
print("Accepted frames:", accepted_count)
print("Rejected frames:", rejected_count)
print("Successfully processed:", processed_count)
print("Output image size:", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Output format: PNG")
print("Output directory:", OUTPUT_DIR)
