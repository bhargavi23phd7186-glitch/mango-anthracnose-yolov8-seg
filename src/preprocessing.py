
import cv2
import numpy as np
import os


def preprocess_image(
    image,
    target_size=(640, 640),
    median_kernel=3,
    clahe_clip_limit=2.0,
    clahe_tile_grid=(8, 8)
):
    """
    Preprocess a mango leaf image for YOLOv8-Seg.

    Processing stages:
        1. Aspect-ratio-preserving resizing with padding.
        2. Median filtering for noise reduction.
        3. CLAHE on the L channel in CIELAB colour space.

    Args:
        image: Input RGB image (NumPy array).
        target_size: Target dimensions (width, height).
        median_kernel: Median filter kernel size (odd integer).
        clahe_clip_limit: CLAHE contrast limit.
        clahe_tile_grid: CLAHE tile grid dimensions.

    Returns:
        Preprocessed RGB image with uint8 data type.
    """

    # --------------------------------------------------
    # Validate input image
    # --------------------------------------------------

    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel RGB image.")

    if median_kernel < 1 or median_kernel % 2 == 0:
        raise ValueError(
            "median_kernel must be a positive odd number."
        )

    if len(target_size) != 2:
        raise ValueError(
            "target_size must contain width and height."
        )

    target_width, target_height = target_size

    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            "Target dimensions must be positive."
        )

    if clahe_clip_limit <= 0:
        raise ValueError(
            "clahe_clip_limit must be positive."
        )

    if (
        len(clahe_tile_grid) != 2
        or clahe_tile_grid[0] <= 0
        or clahe_tile_grid[1] <= 0
    ):
        raise ValueError(
            "clahe_tile_grid dimensions must be positive."
        )

    # --------------------------------------------------
    # Step 1: Resize while preserving aspect ratio
    # --------------------------------------------------

    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = max(
        1, int(round(original_width * scale))
    )

    new_height = max(
        1, int(round(original_height * scale))
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Add padding to obtain the target dimensions

    padded = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    padded[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    # --------------------------------------------------
    # Step 2: Median filtering
    # --------------------------------------------------

    denoised = cv2.medianBlur(
        padded,
        median_kernel
    )

    # --------------------------------------------------
    # Step 3: CLAHE on L channel in CIELAB colour space
    # --------------------------------------------------

    lab_image = cv2.cvtColor(
        denoised,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_grid
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    # --------------------------------------------------
    # Step 4: Output image
    # --------------------------------------------------

    return enhanced_rgb


# ------------------------------------------------------
# Technical verification of preprocessing output
# ------------------------------------------------------

if __name__ == "__main__":

    input_path = "examples/input_leaf.jpg"
    output_path = "results/preprocessed_leaf.png"

    image_bgr = cv2.imread(input_path)

    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not load input image: {input_path}"
        )

    # OpenCV reads images in BGR; convert to RGB

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    processed_image = preprocess_image(
        image_rgb
    )

    # --------------------------------------------------
    # Verify output dimensions and data type
    # --------------------------------------------------

    print("\n--- Preprocessing Verification ---")

    print("Output shape:", processed_image.shape)
    print("Output data type:", processed_image.dtype)
    print("Minimum intensity:", processed_image.min())
    print("Maximum intensity:", processed_image.max())

    assert processed_image.shape == (640, 640, 3)

    assert processed_image.dtype == np.uint8

    # --------------------------------------------------
    # Save processed image
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    processed_bgr = cv2.cvtColor(
        processed_image,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        output_path,
        processed_bgr
    )

    if not success:
        raise IOError(
            "Failed to save preprocessed image."
        )

    print("\nPreprocessing completed successfully.")
    print("Saved output:", output_path)
