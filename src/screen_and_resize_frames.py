
import cv2
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "dataset" / "extracted_frames"
OUTPUT_DIR = ROOT / "dataset" / "processed_frames"

TARGET_SIZE = (640, 640)

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")


# ==========================================
# VALIDATION
# ==========================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Extracted frames directory does not exist: {INPUT_DIR}"
    )

if not INPUT_DIR.is_dir():
    raise NotADirectoryError(
        f"Expected a directory: {INPUT_DIR}"
    )

if (
    len(TARGET_SIZE) != 2
    or any(type(size) is not int for size in TARGET_SIZE)
    or TARGET_SIZE[0] <= 0
    or TARGET_SIZE[1] <= 0
):
    raise ValueError(
        "TARGET_SIZE must contain positive integer width and height."
    )


# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FIND EXTRACTED FRAMES
# ==========================================

image_files = sorted([
    file for file in INPUT_DIR.rglob("*")
    if file.is_file()
    and file.suffix.lower() in SUPPORTED_FORMATS
])

if not image_files:
    raise FileNotFoundError(
        f"No extracted images found in: {INPUT_DIR}"
    )


# ==========================================
# FRAME SCREENING AND RESIZING
# ==========================================

accepted_count = 0
rejected_count = 0
failed_count = 0
unprocessed_count = 0

print("Total extracted frames:", len(image_files))

print("\nScreening controls:")
print("Y = Accept frame")
print("N = Reject frame")
print("Q = Quit screening")


for index, image_path in enumerate(image_files):

    # --------------------------------------
    # Read image
    # --------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        print("Unreadable image:", image_path.name)

        failed_count += 1
        continue

    # --------------------------------------
    # Display image for manual quality review
    # --------------------------------------

    cv2.imshow("Frame Quality Screening", image)

    print("\nReviewing:", image_path.name)

    key = cv2.waitKey(0) & 0xFF

    # --------------------------------------
    # Quit screening
    # --------------------------------------

    if key == ord("q"):

        print("Screening stopped by user.")

        unprocessed_count = (
            len(image_files) - index - 1
        )

        break

    # --------------------------------------
    # Reject frame
    # --------------------------------------

    elif key == ord("n"):

        rejected_count += 1

        print("Rejected:", image_path.name)

        continue

    # --------------------------------------
    # Accept frame
    # --------------------------------------

    elif key == ord("y"):

        # Stretch resize to 640 x 640
        resized = cv2.resize(
            image,
            TARGET_SIZE,
            interpolation=cv2.INTER_LINEAR
        )

        # Preserve video-specific folder structure
        relative_path = image_path.relative_to(INPUT_DIR)

        output_path = (
            OUTPUT_DIR
            / relative_path.parent
            / f"{image_path.stem}.png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save accepted frame as lossless PNG
        success = cv2.imwrite(
            str(output_path),
            resized
        )

        if success:

            accepted_count += 1

            print("Accepted and saved:", output_path)

        else:

            failed_count += 1

            print("ERROR: Could not save:", output_path)

    # --------------------------------------
    # Invalid key
    # --------------------------------------

    else:

        print("Invalid key. Frame skipped.")

        failed_count += 1


# ==========================================
# CLOSE WINDOWS
# ==========================================

cv2.destroyAllWindows()


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n===================================")
print("FRAME SCREENING COMPLETED")
print("===================================")

print("Total frames found:", len(image_files))
print("Accepted frames:", accepted_count)
print("Rejected frames:", rejected_count)
print("Failed/skipped frames:", failed_count)
print("Unprocessed frames:", unprocessed_count)

print(
    "Target dimensions:",
    TARGET_SIZE[0], "x", TARGET_SIZE[1]
)

print("Resize method: Stretch")
print("Output format: PNG")

print("Output directory:", OUTPUT_DIR)
