
from pathlib import Path
import json
import csv
import yaml
from ultralytics import YOLO

# ==========================================================
# YOLOv8n-Seg Independent Test Evaluation
# Mango Anthracnose Instance Segmentation
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

DATA_YAML = ROOT / "dataset" / "mango_dataset" / "data.yaml"

MODEL_PATH = (
    ROOT
    / "runs"
    / "mango_segmentation"
    / "mango_anthracnose_yolov8n_seg"
    / "weights"
    / "best.pt"
)

IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = "cpu"

PROJECT_DIR = ROOT / "evaluation_results"
RUN_NAME = "independent_test_evaluation"

# ----------------------------------------------------------
# Validate required files
# ----------------------------------------------------------

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATA_YAML}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

# ----------------------------------------------------------
# Load dataset configuration
# ----------------------------------------------------------

with open(DATA_YAML, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

if not isinstance(data, dict):
    raise ValueError(
        "Invalid dataset YAML configuration."
    )

if "test" not in data:
    raise ValueError(
        "Dataset YAML must contain a 'test' split."
    )

if "names" not in data:
    raise ValueError(
        "Dataset YAML must contain class names."
    )

# ----------------------------------------------------------
# Validate class names
# ----------------------------------------------------------

names = data["names"]

if isinstance(names, dict):
    class_names = [
        names[key] for key in sorted(names.keys())
    ]
else:
    class_names = list(names)

if data.get("nc") != len(class_names):
    raise ValueError(
        f"Mismatch between nc ({data.get('nc')}) "
        f"and number of classes ({len(class_names)})."
    )

required_classes = {"leaf", "anthracnose"}

if set(class_names) != required_classes:
    raise ValueError(
        "Unexpected class names.\n"
        f"Expected: {required_classes}\n"
        f"Found: {set(class_names)}"
    )

# ----------------------------------------------------------
# Resolve dataset root
# ----------------------------------------------------------

yaml_dir = DATA_YAML.parent

configured_root = data.get("path", yaml_dir)
configured_root = Path(configured_root)

if configured_root.is_absolute():

    dataset_root = configured_root.resolve()

else:

    candidate_1 = (
        yaml_dir / configured_root
    ).resolve()

    candidate_2 = (
        ROOT / configured_root
    ).resolve()

    if candidate_1.exists():
        dataset_root = candidate_1

    elif candidate_2.exists():
        dataset_root = candidate_2

    else:
        dataset_root = candidate_1

# ----------------------------------------------------------
# Resolve test directory
# ----------------------------------------------------------

test_path = Path(data["test"])

if test_path.is_absolute():

    test_dir = test_path.resolve()

else:

    test_candidates = [
        (dataset_root / test_path).resolve(),
        (yaml_dir / test_path).resolve(),
        (ROOT / test_path).resolve(),
    ]

    test_dir = next(
        (
            candidate
            for candidate in test_candidates
            if candidate.exists()
        ),
        test_candidates[0],
    )

# ----------------------------------------------------------
# Validate test directory
# ----------------------------------------------------------

if not test_dir.exists():
    raise FileNotFoundError(
        f"Test path does not exist:\n{test_dir}"
    )

if not test_dir.is_dir():
    raise NotADirectoryError(
        f"Test path is not a directory:\n{test_dir}"
    )

# ----------------------------------------------------------
# Count test images
# ----------------------------------------------------------

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

test_images = [
    file
    for file in test_dir.rglob("*")
    if file.is_file()
    and file.suffix.lower() in valid_extensions
]

if len(test_images) == 0:
    raise RuntimeError(
        f"No test images found in:\n{test_dir}"
    )

# ----------------------------------------------------------
# Display configuration
# ----------------------------------------------------------

print("=" * 65)
print("INDEPENDENT TEST EVALUATION")
print("=" * 65)

print(f"Model:             {MODEL_PATH}")
print(f"Dataset YAML:      {DATA_YAML}")
print(f"Test directory:    {test_dir}")
print(f"Number of images:  {len(test_images)}")
print(f"Classes:           {class_names}")
print(f"Image size:        {IMAGE_SIZE}")
print(f"Batch size:        {BATCH_SIZE}")
print(f"Device:            {DEVICE}")

# ----------------------------------------------------------
# Load trained model
# ----------------------------------------------------------

model = YOLO(str(MODEL_PATH))

# ----------------------------------------------------------
# Create evaluation project directory
# ----------------------------------------------------------

PROJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# Evaluate using independent test split
# ----------------------------------------------------------

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    plots=True,
    save_json=True,
    verbose=True,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    exist_ok=False,
)

# ----------------------------------------------------------
# Extract detection metrics
# ----------------------------------------------------------

box_metrics = metrics.box

detection_precision = float(box_metrics.mp)
detection_recall = float(box_metrics.mr)

if (
    detection_precision + detection_recall
) > 0:

    detection_f1 = (
        2
        * detection_precision
        * detection_recall
        / (
            detection_precision
            + detection_recall
        )
    )

else:

    detection_f1 = 0.0

box_map50 = float(box_metrics.map50)
box_map50_95 = float(box_metrics.map)

# ----------------------------------------------------------
# Extract segmentation metrics
# ----------------------------------------------------------

seg_metrics = metrics.seg

mask_map50 = float(seg_metrics.map50)
mask_map50_95 = float(seg_metrics.map)

# ----------------------------------------------------------
# Identify actual evaluation output directory
# ----------------------------------------------------------

saved_dir = getattr(
    metrics,
    "save_dir",
    None
)

if saved_dir is not None:

    output_dir = Path(saved_dir)

else:

    # Fallback if the installed Ultralytics version
    # does not expose metrics.save_dir.

    output_dir = PROJECT_DIR / RUN_NAME

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# Compile results
# ----------------------------------------------------------

results = {
    "model_path": str(MODEL_PATH),
    "dataset_yaml": str(DATA_YAML),
    "test_directory": str(test_dir),
    "test_image_count": len(test_images),
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "device": DEVICE,

    "detection_precision": detection_precision,
    "detection_recall": detection_recall,
    "detection_f1_score": detection_f1,

    "box_map50": box_map50,
    "box_map50_95": box_map50_95,

    "mask_map50": mask_map50,
    "mask_map50_95": mask_map50_95,
}

# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("INDEPENDENT TEST RESULTS")
print("=" * 65)

print(f"Test images: {len(test_images)}")

print("\nDetection Metrics")
print("-" * 40)

print(
    f"Detection Precision: "
    f"{detection_precision:.4f}"
)

print(
    f"Detection Recall:    "
    f"{detection_recall:.4f}"
)

print(
    f"Detection F1-score:  "
    f"{detection_f1:.4f}"
)

print(
    f"Box mAP@0.5:         "
    f"{box_map50:.4f}"
)

print(
    f"Box mAP@0.5-0.95:    "
    f"{box_map50_95:.4f}"
)

print("\nSegmentation Metrics")
print("-" * 40)

print(
    f"Mask mAP@0.5:        "
    f"{mask_map50:.4f}"
)

print(
    f"Mask mAP@0.5-0.95:   "
    f"{mask_map50_95:.4f}"
)

# ----------------------------------------------------------
# Save JSON report
# ----------------------------------------------------------

json_path = output_dir / "test_metrics.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

# ----------------------------------------------------------
# Save CSV report
# ----------------------------------------------------------

csv_path = output_dir / "test_metrics.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Metric", "Value"]
    )

    for key, value in results.items():

        if isinstance(
            value,
            (int, float)
        ):

            writer.writerow(
                [key, value]
            )

        else:

            writer.writerow(
                [key, str(value)]
            )

# ----------------------------------------------------------
# Final summary
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print(
    f"\nActual evaluation output directory:\n"
    f"{output_dir}"
)

print(
    f"\nJSON report:\n{json_path}"
)

print(
    f"\nCSV report:\n{csv_path}"
)

print(
    "\nUltralytics evaluation plots, including "
    "confusion matrices and metric curves, "
    "are saved in the evaluation run directory."
)
