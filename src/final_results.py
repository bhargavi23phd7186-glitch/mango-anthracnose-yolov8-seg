
# ============================================================
# final_results.py
# Independent image-level disease classification evaluation
# for Mango Anthracnose YOLOv8-Seg
# ============================================================

from pathlib import Path
import json
import csv
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report,
)


# ============================================================
# 1. PROJECT AND DATASET PATHS
# ============================================================

# Original dataset location provided by the user
DATASET_ROOT = Path(r"C:/Users/admin/Downloads/mango.yolov8")

DATA_YAML = DATASET_ROOT / "data.yaml"

# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------
# IMPORTANT:
# Change PROJECT_ROOT if your trained model is stored elsewhere.
#
# Expected example:
# PROJECT_ROOT/
#   runs/
#     mango_segmentation/
#       mango_anthracnose_yolov8n_seg/
#         weights/
#           best.pt
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "mango_segmentation"
    / "mango_anthracnose_yolov8n_seg"
    / "weights"
    / "best.pt"
)


# Output directory
OUTPUT_DIR = PROJECT_ROOT / "results" / "image_level_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EVALUATION SETTINGS
# ============================================================

IMAGE_SIZE = 640

# Low confidence threshold to retain disease detections
INFERENCE_CONFIDENCE = 0.001

# Binary image-level decision threshold
CLASSIFICATION_THRESHOLD = 0.25

DEVICE = "cpu"

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 3. VALIDATION
# ============================================================

def validate_paths():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_ROOT}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found:\n{MODEL_PATH}\n\n"
            "Check the actual training run directory and correct "
            "MODEL_PATH."
        )

    print("Dataset directory:", DATASET_ROOT)
    print("Dataset YAML:", DATA_YAML)
    print("Model weights:", MODEL_PATH)


# ============================================================
# 4. LOAD DATASET CONFIGURATION
# ============================================================

def load_dataset_config():

    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid data.yaml configuration.")

    if "names" not in data:
        raise ValueError("data.yaml must contain 'names'.")

    names = data["names"]

    # Convert names to a dictionary consistently
    if isinstance(names, list):

        class_names = {
            index: name
            for index, name in enumerate(names)
        }

    elif isinstance(names, dict):

        class_names = {
            int(index): name
            for index, name in names.items()
        }

    else:

        raise ValueError(
            "Class names must be a list or dictionary."
        )

    print("\nDataset class mapping:")

    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    # Identify classes dynamically
    leaf_class_id = None
    anthracnose_class_id = None

    for class_id, class_name in class_names.items():

        normalized_name = str(class_name).lower().strip()

        if normalized_name == "leaf":
            leaf_class_id = class_id

        elif normalized_name == "anthracnose":
            anthracnose_class_id = class_id

    if leaf_class_id is None:
        raise ValueError(
            "The 'leaf' class was not found in data.yaml."
        )

    if anthracnose_class_id is None:
        raise ValueError(
            "The 'anthracnose' class was not found in data.yaml."
        )

    return data, class_names, leaf_class_id, anthracnose_class_id


# ============================================================
# 5. VALIDATE MODEL CLASS MAPPING
# ============================================================

def validate_model_class_mapping(
    model,
    dataset_class_names
):

    model_names = model.names

    # Convert model names to dictionary
    if isinstance(model_names, list):

        model_class_names = {
            index: name
            for index, name in enumerate(model_names)
        }

    elif isinstance(model_names, dict):

        model_class_names = {
            int(index): name
            for index, name in model_names.items()
        }

    else:

        raise ValueError(
            "Unable to read model class names."
        )

    print("\nModel class mapping:")

    for class_id, class_name in model_class_names.items():
        print(f"  {class_id}: {class_name}")

    # Compare mappings
    normalized_dataset_mapping = {
        int(class_id): str(class_name).lower().strip()
        for class_id, class_name
        in dataset_class_names.items()
    }

    normalized_model_mapping = {
        int(class_id): str(class_name).lower().strip()
        for class_id, class_name
        in model_class_names.items()
    }

    if normalized_dataset_mapping != normalized_model_mapping:

        raise ValueError(
            "\nERROR: Model and dataset class mappings do not match!\n\n"
            f"Dataset mapping:\n{normalized_dataset_mapping}\n\n"
            f"Model mapping:\n{normalized_model_mapping}\n\n"
            "Evaluation stopped to prevent incorrect class interpretation."
        )

    print("\nClass mapping validation PASSED.")

    return model_class_names


# ============================================================
# 6. RESOLVE TEST IMAGE DIRECTORY
# ============================================================

def resolve_dataset_path(path_value):

    path = Path(path_value)

    if path.is_absolute():
        return path

    # Dataset YAML paths may be relative to dataset root
    return DATASET_ROOT / path


def get_test_image_directory(data):

    if "test" not in data:
        raise ValueError(
            "The data.yaml file does not contain a test path."
        )

    test_path = resolve_dataset_path(data["test"])

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{test_path}"
        )

    return test_path


# ============================================================
# 7. RESOLVE LABEL DIRECTORY
# ============================================================

def get_label_directory(test_image_dir):

    # Typical YOLO dataset structure:
    #
    # test/
    #   images/
    #   labels/
    #
    # or:
    #
    # test/
    #   image1.jpg
    #   image1.txt

    if test_image_dir.name.lower() == "images":

        label_dir = test_image_dir.parent / "labels"

    else:

        label_dir = test_image_dir.parent / "labels"

    if not label_dir.exists():

        raise FileNotFoundError(
            f"Label directory not found:\n{label_dir}\n\n"
            "Check your dataset structure."
        )

    return label_dir


# ============================================================
# 8. VALIDATE YOLO SEGMENTATION LABELS
# ============================================================

def read_ground_truth_label(
    label_path,
    leaf_class_id,
    anthracnose_class_id
):

    if not label_path.exists():

        raise FileNotFoundError(
            f"Missing annotation file:\n{label_path}"
        )

    has_leaf = False
    has_anthracnose = False

    with open(label_path, "r", encoding="utf-8") as file:

        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Empty annotation file is not assumed healthy automatically
        if not line:
            continue

        parts = line.split()

        if len(parts) < 7:

            raise ValueError(
                f"Invalid polygon annotation at:\n"
                f"{label_path}\n"
                f"Line: {line_number}"
            )

        try:

            class_id = int(parts[0])

            coordinates = [
                float(value)
                for value in parts[1:]
            ]

        except ValueError:

            raise ValueError(
                f"Invalid numeric annotation at:\n"
                f"{label_path}\n"
                f"Line: {line_number}"
            )

        # Polygon must contain at least 3 points
        if len(coordinates) < 6:

            raise ValueError(
                f"Polygon has fewer than 3 points:\n"
                f"{label_path}\n"
                f"Line: {line_number}"
            )

        # Polygon coordinates must be paired
        if len(coordinates) % 2 != 0:

            raise ValueError(
                f"Invalid polygon coordinate count:\n"
                f"{label_path}\n"
                f"Line: {line_number}"
            )

        # Validate normalized coordinates
        for coordinate in coordinates:

            if (
                not np.isfinite(coordinate)
                or coordinate < 0
                or coordinate > 1
            ):

                raise ValueError(
                    f"Invalid normalized coordinate:\n"
                    f"{label_path}\n"
                    f"Line: {line_number}"
                )

        if class_id == leaf_class_id:

            has_leaf = True

        elif class_id == anthracnose_class_id:

            has_anthracnose = True

    # This dataset convention requires leaf polygons
    if not has_leaf:

        raise ValueError(
            f"No leaf annotation found in:\n{label_path}\n\n"
            "Please check whether this image has been annotated "
            "according to the expected dataset convention."
        )

    # Healthy = leaf present, anthracnose absent
    # Diseased = anthracnose polygon present
    ground_truth = 1 if has_anthracnose else 0

    return ground_truth


# ============================================================
# 9. COLLECT TEST IMAGES
# ============================================================

def collect_test_images(test_image_dir):

    image_paths = sorted(
        [
            path
            for path in test_image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if len(image_paths) == 0:

        raise RuntimeError(
            f"No test images found in:\n{test_image_dir}"
        )

    print("\nNumber of test images:", len(image_paths))

    return image_paths


# ============================================================
# 10. IMAGE-LEVEL DISEASE SCORE
# ============================================================

def get_anthracnose_score(
    result,
    anthracnose_class_id
):

    # No detections
    if result.boxes is None or len(result.boxes) == 0:

        return 0.0

    class_ids = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    disease_confidences = confidences[
        class_ids == anthracnose_class_id
    ]

    if len(disease_confidences) == 0:

        return 0.0

    # Image-level score:
    # maximum anthracnose detection confidence
    return float(
        np.max(disease_confidences)
    )


# ============================================================
# 11. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("MANGO ANTHRACNOSIS IMAGE-LEVEL EVALUATION")
    print("=" * 70)

    # Validate project paths
    validate_paths()

    # Load dataset configuration
    (
        data,
        dataset_class_names,
        leaf_class_id,
        anthracnose_class_id,
    ) = load_dataset_config()

    # Load model
    print("\nLoading trained model...")

    model = YOLO(str(MODEL_PATH))

    # Validate class mapping
    validate_model_class_mapping(
        model,
        dataset_class_names
    )

    # Resolve test directory
    test_image_dir = get_test_image_directory(data)

    # Resolve labels
    label_dir = get_label_directory(test_image_dir)

    print("\nTest images:", test_image_dir)
    print("Test labels:", label_dir)

    # Collect images
    image_paths = collect_test_images(test_image_dir)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    ground_truths = []
    disease_scores = []
    predicted_labels = []

    records = []

    # --------------------------------------------------------
    # Evaluate every test image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_paths, start=1):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Evaluating: {image_path.name}"
        )

        # Corresponding label
        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        # Ground truth from annotation
        ground_truth = read_ground_truth_label(
            label_path,
            leaf_class_id,
            anthracnose_class_id
        )

        # Model inference
        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=INFERENCE_CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        # Image-level disease score
        disease_score = get_anthracnose_score(
            result,
            anthracnose_class_id
        )

        # Binary classification threshold
        predicted_label = (
            1
            if disease_score >= CLASSIFICATION_THRESHOLD
            else 0
        )

        ground_truths.append(ground_truth)
        disease_scores.append(disease_score)
        predicted_labels.append(predicted_label)

        records.append(
            {
                "image_name": image_path.name,
                "ground_truth": ground_truth,
                "ground_truth_label": (
                    "Diseased"
                    if ground_truth == 1
                    else "Healthy"
                ),
                "disease_score": disease_score,
                "predicted_label": predicted_label,
                "predicted_class": (
                    "Diseased"
                    if predicted_label == 1
                    else "Healthy"
                ),
                "correct": (
                    ground_truth == predicted_label
                ),
            }
        )

        print(
            f"Ground truth: "
            f"{'Diseased' if ground_truth else 'Healthy'}"
        )

        print(
            f"Disease score: {disease_score:.6f}"
        )

        print(
            f"Prediction: "
            f"{'Diseased' if predicted_label else 'Healthy'}"
        )

    # Convert to arrays
    y_true = np.array(ground_truths)
    y_score = np.array(disease_scores)
    y_pred = np.array(predicted_labels)

    # ========================================================
    # 12. CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0.0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # 13. ROC AND AUC
    # ========================================================

    if len(np.unique(y_true)) < 2:

        raise ValueError(
            "ROC/AUC cannot be calculated because the test set "
            "contains only one ground-truth class."
        )

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        y_score
    )

    auc_value = roc_auc_score(
        y_true,
        y_score
    )

    # ========================================================
    # 14. PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL IMAGE-LEVEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"Test images: {len(y_true)}")

    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TP: {TP}")

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"AUC:         {auc_value:.4f}")

    print(
        f"\nInference confidence threshold: "
        f"{INFERENCE_CONFIDENCE}"
    )

    print(
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD}"
    )

    print(
        "\nImage-level score: maximum anthracnose "
        "detection confidence"
    )

    # ========================================================
    # 15. SAVE CONFUSION MATRIX
    # ========================================================

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Image-Level Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")

    plt.xticks(
        [0, 1],
        ["Healthy", "Diseased"]
    )

    plt.yticks(
        [0, 1],
        ["Healthy", "Diseased"]
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    confusion_matrix_png = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    confusion_matrix_pdf = (
        OUTPUT_DIR
        / "confusion_matrix.pdf"
    )

    plt.savefig(
        confusion_matrix_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        confusion_matrix_pdf,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 16. SAVE ROC CURVE
    # ========================================================

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {auc_value:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Image-Level Disease Classification")

    plt.legend(loc="lower right")

    plt.grid(True)

    plt.tight_layout()

    roc_png = OUTPUT_DIR / "roc_curve.png"
    roc_pdf = OUTPUT_DIR / "roc_curve.pdf"

    plt.savefig(
        roc_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        roc_pdf,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # 17. SAVE PREDICTIONS CSV
    # ========================================================

    predictions_csv = (
        OUTPUT_DIR
        / "image_level_predictions.csv"
    )

    with open(
        predictions_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()

        writer.writerows(records)

    # ========================================================
    # 18. SAVE METRICS
    # ========================================================

    metrics = {

        "dataset": str(DATASET_ROOT),

        "model": str(MODEL_PATH),

        "test_images": int(len(y_true)),

        "confidence_threshold": (
            INFERENCE_CONFIDENCE
        ),

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "score_definition": (
            "Maximum anthracnose detection confidence "
            "per image"
        ),

        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "sensitivity": float(recall),

        "specificity": float(specificity),

        "f1_score": float(f1),

        "auc": float(auc_value),
    }

    metrics_json = (
        OUTPUT_DIR
        / "image_level_metrics.json"
    )

    with open(
        metrics_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    metrics_csv = (
        OUTPUT_DIR
        / "image_level_metrics.csv"
    )

    with open(
        metrics_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Metric", "Value"]
        )

        for key, value in metrics.items():

            writer.writerow(
                [key, value]
            )

    # ========================================================
    # 19. CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Healthy",
            "Diseased"
        ],
        zero_division=0
    )

    report_txt = (
        OUTPUT_DIR
        / "classification_report.txt"
    )

    with open(
        report_txt,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nClassification report:")
    print(report)

    # ========================================================
    # 20. COMPLETION
    # ========================================================

    print("\nEvaluation completed successfully.")

    print("\nFiles saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("- image_level_predictions.csv")
    print("- image_level_metrics.csv")
    print("- image_level_metrics.json")
    print("- classification_report.txt")
    print("- confusion_matrix.png")
    print("- confusion_matrix.pdf")
    print("- roc_curve.png")
    print("- roc_curve.pdf")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
