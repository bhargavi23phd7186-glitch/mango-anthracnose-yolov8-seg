
from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================================
# Mango Anthracnose Disease Severity Estimation
# YOLOv8-Seg Instance Segmentation
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

MODEL_PATH = (
    ROOT
    / "runs"
    / "mango_segmentation"
    / "mango_anthracnose_yolov8n_seg"
    / "weights"
    / "best.pt"
)

TEST_DIR = (
    ROOT
    / "dataset"
    / "mango_dataset"
    / "test"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "severity_results"
)

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cpu"

# Severity thresholds
T1 = 10.0
T2 = 25.0

# ----------------------------------------------------------
# Validate paths
# ----------------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test image directory not found:\n{TEST_DIR}"
    )

if not TEST_DIR.is_dir():
    raise NotADirectoryError(
        f"Test path is not a directory:\n{TEST_DIR}"
    )

# ----------------------------------------------------------
# Find test images
# ----------------------------------------------------------

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

image_paths = sorted(
    [
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]
)

if not image_paths:
    raise RuntimeError(
        f"No images found in:\n{TEST_DIR}"
    )

# ----------------------------------------------------------
# Create output folders
# ----------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"

ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------

model = YOLO(str(MODEL_PATH))

# ----------------------------------------------------------
# Validate model class names
# ----------------------------------------------------------

model_names = model.names

if isinstance(model_names, dict):
    names = {
        int(class_id): class_name.lower()
        for class_id, class_name
        in model_names.items()
    }
else:
    names = {
        class_id: class_name.lower()
        for class_id, class_name
        in enumerate(model_names)
    }

print("Model classes:", names)

leaf_class_ids = [
    class_id
    for class_id, class_name in names.items()
    if class_name == "leaf"
]

disease_class_ids = [
    class_id
    for class_id, class_name in names.items()
    if class_name == "anthracnose"
]

if not leaf_class_ids:
    raise ValueError(
        "The model does not contain a 'leaf' class."
    )

if not disease_class_ids:
    raise ValueError(
        "The model does not contain an 'anthracnose' class."
    )

LEAF_CLASS_ID = leaf_class_ids[0]
DISEASE_CLASS_ID = disease_class_ids[0]

# ----------------------------------------------------------
# Severity classification
# ----------------------------------------------------------

def classify_severity(severity_percent):

    if severity_percent is None:
        return "No leaf detected"

    if severity_percent == 0:
        return "Healthy"

    elif severity_percent <= T1:
        return "Mild"

    elif severity_percent <= T2:
        return "Moderate"

    else:
        return "Severe"


# ----------------------------------------------------------
# Calculate mask areas
# ----------------------------------------------------------

def calculate_mask_areas(result):

    # No segmentation masks
    if result.masks is None:
        return None, None, None

    masks = result.masks.data

    if masks is None or len(masks) == 0:
        return None, None, None

    masks = masks.cpu().numpy()

    classes = result.boxes.cls.cpu().numpy().astype(int)

    # Mask dimensions
    mask_height, mask_width = masks.shape[1:]

    # ------------------------------------------------------
    # Build combined leaf mask
    # ------------------------------------------------------

    leaf_masks = masks[
        np.isin(classes, leaf_class_ids)
    ]

    if len(leaf_masks) > 0:

        leaf_union = np.any(
            leaf_masks > 0.5,
            axis=0
        )

    else:

        leaf_union = np.zeros(
            (mask_height, mask_width),
            dtype=bool
        )

    # ------------------------------------------------------
    # Build combined anthracnose mask
    # ------------------------------------------------------

    disease_masks = masks[
        np.isin(classes, disease_class_ids)
    ]

    if len(disease_masks) > 0:

        disease_union = np.any(
            disease_masks > 0.5,
            axis=0
        )

    else:

        disease_union = np.zeros(
            (mask_height, mask_width),
            dtype=bool
        )

    # ------------------------------------------------------
    # Calculate leaf area
    # ------------------------------------------------------

    leaf_area_pixels = int(
        np.count_nonzero(leaf_union)
    )

    # ------------------------------------------------------
    # Restrict disease mask to leaf area
    # ------------------------------------------------------

    disease_within_leaf = (
        disease_union & leaf_union
    )

    disease_area_pixels = int(
        np.count_nonzero(disease_within_leaf)
    )

    # ------------------------------------------------------
    # Calculate severity
    # ------------------------------------------------------

    if leaf_area_pixels == 0:

        severity_percent = None

    else:

        severity_percent = (
            disease_area_pixels
            / leaf_area_pixels
        ) * 100.0

    return (
        leaf_area_pixels,
        disease_area_pixels,
        severity_percent,
    )


# ----------------------------------------------------------
# Create visualization
# ----------------------------------------------------------

def create_annotated_image(
    image,
    result,
    severity_percent,
    category,
):

    annotated = image.copy()

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        mask_height, mask_width = masks.shape[1:]

        leaf_masks = masks[
            np.isin(classes, leaf_class_ids)
        ]

        disease_masks = masks[
            np.isin(classes, disease_class_ids)
        ]

        if len(leaf_masks) > 0:

            leaf_union = np.any(
                leaf_masks > 0.5,
                axis=0
            )

        else:

            leaf_union = np.zeros(
                (mask_height, mask_width),
                dtype=bool
            )

        if len(disease_masks) > 0:

            disease_union = np.any(
                disease_masks > 0.5,
                axis=0
            )

        else:

            disease_union = np.zeros(
                (mask_height, mask_width),
                dtype=bool
            )

        disease_union = (
            disease_union & leaf_union
        )

        # Resize masks to original image dimensions
        if (
            leaf_union.shape
            != image.shape[:2]
        ):

            leaf_union = cv2.resize(
                leaf_union.astype(np.uint8),
                (
                    image.shape[1],
                    image.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            disease_union = cv2.resize(
                disease_union.astype(np.uint8),
                (
                    image.shape[1],
                    image.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # --------------------------------------------------
        # Leaf overlay: green
        # --------------------------------------------------

        leaf_color = np.array(
            [0, 255, 0],
            dtype=np.uint8
        )

        annotated[leaf_union] = (
            0.5 * annotated[leaf_union]
            + 0.5 * leaf_color
        ).astype(np.uint8)

        # --------------------------------------------------
        # Disease overlay: red
        # --------------------------------------------------

        disease_color = np.array(
            [0, 0, 255],
            dtype=np.uint8
        )

        annotated[disease_union] = (
            0.5 * annotated[disease_union]
            + 0.5 * disease_color
        ).astype(np.uint8)

    # ------------------------------------------------------
    # Add severity text
    # ------------------------------------------------------

    if severity_percent is None:

        severity_text = (
            "Severity: No leaf detected"
        )

    else:

        severity_text = (
            f"Severity: {severity_percent:.2f}% "
            f"({category})"
        )

    cv2.putText(
        annotated,
        severity_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


# ----------------------------------------------------------
# Process test images
# ----------------------------------------------------------

records = []

print("\nStarting severity estimation...")
print("=" * 60)

for index, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"[{index}/{len(image_paths)}] "
        f"Processing: {image_path.name}"
    )

    # ------------------------------------------------------
    # Load original image
    # ------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"WARNING: Could not read image: "
            f"{image_path}"
        )

        continue

    # ------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------

    prediction = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    result = prediction[0]

    # ------------------------------------------------------
    # Calculate areas
    # ------------------------------------------------------

    (
        leaf_area_pixels,
        disease_area_pixels,
        severity_percent,
    ) = calculate_mask_areas(result)

    # ------------------------------------------------------
    # Assign severity class
    # ------------------------------------------------------

    severity_category = classify_severity(
        severity_percent
    )

    # ------------------------------------------------------
    # Save annotated image
    # ------------------------------------------------------

    annotated = create_annotated_image(
        image=image,
        result=result,
        severity_percent=severity_percent,
        category=severity_category,
    )

    output_image_path = (
        ANNOTATED_DIR
        / f"{image_path.stem}_severity.png"
    )

    image_saved = cv2.imwrite(
        str(output_image_path),
        annotated
    )

    if not image_saved:

        print(
            "WARNING: Failed to save annotated image:"
            f"{output_image_path}"
        )

    # ------------------------------------------------------
    # Record results
    # ------------------------------------------------------

    record = {
        "image_name": image_path.name,
        "image_path": str(image_path),

        "leaf_area_pixels": (
            leaf_area_pixels
        ),

        "anthracnose_area_pixels": (
            disease_area_pixels
        ),

        "disease_severity_percent": (
            severity_percent
        ),

        "severity_category": (
            severity_category
        ),

        "annotated_image": (
            str(output_image_path)
        ),
    }

    records.append(record)

    # ------------------------------------------------------
    # Print per-image results
    # ------------------------------------------------------

    print(
        f"Leaf area: "
        f"{leaf_area_pixels}"
    )

    print(
        f"Anthracnose area: "
        f"{disease_area_pixels}"
    )

    if severity_percent is not None:

        print(
            f"Disease severity: "
            f"{severity_percent:.2f}%"
        )

    else:

        print(
            "Disease severity: "
            "Not calculated (leaf not detected)"
        )

    print(
        f"Classification: "
        f"{severity_category}"
    )

    print("-" * 60)

# ----------------------------------------------------------
# Save CSV results
# ----------------------------------------------------------

csv_path = (
    OUTPUT_DIR
    / "severity_results.csv"
)

fieldnames = [
    "image_name",
    "image_path",
    "leaf_area_pixels",
    "anthracnose_area_pixels",
    "disease_severity_percent",
    "severity_category",
    "annotated_image",
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(records)

# ----------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------

valid_severity_values = [
    record["disease_severity_percent"]
    for record in records
    if record["disease_severity_percent"]
    is not None
]

category_counts = {
    "Healthy": 0,
    "Mild": 0,
    "Moderate": 0,
    "Severe": 0,
    "No leaf detected": 0,
}

for record in records:

    category = record["severity_category"]

    if category in category_counts:

        category_counts[category] += 1

# ----------------------------------------------------------
# Print summary
# ----------------------------------------------------------

print("\n")
print("=" * 60)
print("SEVERITY ESTIMATION SUMMARY")
print("=" * 60)

print(
    f"Images processed: "
    f"{len(records)}"
)

print(
    f"Images with leaf detection: "
    f"{len(valid_severity_values)}"
)

print("\nSeverity Category Distribution")
print("-" * 40)

for category, count in category_counts.items():

    print(
        f"{category:<20}: {count}"
    )

if valid_severity_values:

    print("\nSeverity Statistics")
    print("-" * 40)

    print(
        f"Mean severity: "
        f"{np.mean(valid_severity_values):.2f}%"
    )

    print(
        f"Minimum severity: "
        f"{np.min(valid_severity_values):.2f}%"
    )

    print(
        f"Maximum severity: "
        f"{np.max(valid_severity_values):.2f}%"
    )

# ----------------------------------------------------------
# Final output
# ----------------------------------------------------------

print("\n")
print("=" * 60)
print("SEVERITY ESTIMATION COMPLETED")
print("=" * 60)

print(
    f"CSV results:\n{csv_path}"
)

print(
    f"Annotated images:\n{ANNOTATED_DIR}"
)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)
