
import cv2
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

VIDEO_DIR = ROOT / "dataset" / "raw_videos"
FRAME_DIR = ROOT / "dataset" / "extracted_frames"

# --------------------------------------------------
# Frame extraction configuration
# --------------------------------------------------

FRAME_INTERVAL = 15

# Reference metadata reported in the paper
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
REFERENCE_FPS = 58.53

SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]


def test_extraction():

    total_videos = 0
    total_expected = 0
    total_extracted = 0
    passed = 0

    # Check whether the raw video directory exists
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}"
        )

    for video in sorted(VIDEO_DIR.iterdir()):

        if video.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        total_videos += 1

        cap = cv2.VideoCapture(str(video))

        if not cap.isOpened():
            print(f"FAIL: Cannot open {video.name}")
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if frames <= 0 or fps <= 0:
            print(f"FAIL: Invalid video metadata: {video.name}")
            continue

        # Expected frame indices: 0, 15, 30, 45, ...
        expected_indices = list(
            range(0, frames, FRAME_INTERVAL)
        )

        expected = len(expected_indices)

        # Corresponding extracted-frame directory
        folder = FRAME_DIR / video.stem

        images = sorted(
            folder.glob(f"{video.stem}_frame_*.png")
        ) if folder.exists() else []

        actual_indices = []
        readable = True
        dimensions_correct = True

        for image_path in images:

            try:
                index = int(
                    image_path.stem.split("_frame_")[-1]
                )

                actual_indices.append(index)

                img = cv2.imread(str(image_path))

                if img is None:
                    readable = False
                    continue

                h, w = img.shape[:2]

                if w != width or h != height:
                    dimensions_correct = False

            except ValueError:
                readable = False

        # Verify frame sampling sequence
        sampling_correct = (
            sorted(actual_indices) == expected_indices
        )

        # Verify extracted frame count
        count_correct = len(images) == expected

        # Overall test
        test_pass = all([
            sampling_correct,
            count_correct,
            readable,
            dimensions_correct
        ])

        if test_pass:
            passed += 1

        total_expected += expected
        total_extracted += len(images)

        print("\n-----------------------------------")
        print("Video:", video.name)
        print(
            "Duration:",
            round(frames / fps / 60, 2),
            "minutes"
        )
        print("Resolution:", width, "x", height)
        print("FPS:", round(fps, 2))
        print("Expected frames:", expected)
        print("Extracted frames:", len(images))

        print(
            "Sampling:",
            "PASS" if sampling_correct else "FAIL"
        )

        print(
            "Frame count:",
            "PASS" if count_correct else "FAIL"
        )

        print(
            "Image readability:",
            "PASS" if readable else "FAIL"
        )

        print(
            "Resolution consistency:",
            "PASS" if dimensions_correct else "FAIL"
        )

        print(
            "Overall:",
            "PASS" if test_pass else "FAIL"
        )

    # Final summary
    print("\n========== FINAL TEST RESULTS ==========")
    print("Total videos:", total_videos)
    print("Passed videos:", passed)
    print("Failed videos:", total_videos - passed)
    print("Expected extracted frames:", total_expected)
    print("Actual extracted frames:", total_extracted)

    if total_videos > 0 and passed == total_videos:
        print("\nSTEP 1 TEST: PASSED")
    else:
        print("\nSTEP 1 TEST: FAILED")


if __name__ == "__main__":
    test_extraction()
