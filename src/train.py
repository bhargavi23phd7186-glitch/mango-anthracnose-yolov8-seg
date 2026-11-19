from pathlib import Path
import yaml
from ultralytics import YOLO

# ==========================================================
# YOLOv8n-Seg Training
# Mango Anthracnose Instance Segmentation
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "mango_dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

MODEL_NAME = "yolov8n-seg.pt"

EPOCHS = 150
IMAGE_SIZE = 640
BATCH_SIZE = 8

DEVICE = "cpu"
WORKERS = 0

SEED = 42

PROJECT_DIR = ROOT / "runs" / "mango_segmentation"
RUN_NAME = "mango_anthracnose_yolov8n_seg"


# ==========================================================
# Resolve dataset paths
# ==========================================================

def resolve_dataset_path(path_value, dataset_root, yaml_dir):

    path = Path(path_value)

    # Absolute path
    if path.is_absolute():
        return path.resolve()

    # Try dataset root
    candidate_1 = (dataset_root / path).resolve()

    if candidate_1.exists():
        return candidate_1

    # Try YAML directory
    candidate_2 = (yaml_dir / path).resolve()

    if candidate_2.exists():
        return candidate_2

    # Try project root
    candidate_3 = (ROOT / path).resolve()

    if candidate_3.exists():
        return candidate_3

    # Return primary candidate for error reporting
    return candidate_1


# ==========================================================
# Validate data.yaml and dataset paths
# ==========================================================

def validate_dataset():

    print("=" * 60)
    print("DATASET CONFIGURATION CHECK")
    print("=" * 60)

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    # ------------------------------------------------------
    # Load YAML
    # ------------------------------------------------------

    with open(DATA_YAML, "r", encoding="utf-8") as file:

        data = yaml.safe_load(file)

    if not isinstance(data, dict):

        raise ValueError(
            "Invalid data.yaml configuration."
        )

    yaml_dir = DATA_YAML.parent.resolve()

    # ------------------------------------------------------
    # Resolve dataset root
    # ------------------------------------------------------

    configured_path = data.get("path", yaml_dir)

    dataset_root = Path(configured_path)

    if not dataset_root.is_absolute():

        candidate_1 = (yaml_dir / dataset_root).resolve()
        candidate_2 = (ROOT / dataset_root).resolve()

        if candidate_1.exists():

            dataset_root = candidate_1

        elif candidate_2.exists():

            dataset_root = candidate_2

        else:

            dataset_root = candidate_1

    else:

        dataset_root = dataset_root.resolve()

    print(f"\nDataset YAML : {DATA_YAML}")
    print(f"Dataset root: {dataset_root}")

    # ------------------------------------------------------
    # Validate class names
    # ------------------------------------------------------

    names = data.get("names")

    if names is None:

        raise ValueError(
            "The data.yaml file does not contain 'names'."
        )

    if isinstance(names, list):

        class_names = names

    elif isinstance(names, dict):

        class_names = [
            names[key]
            for key in sorted(
                names,
                key=lambda key: int(key)
            )
        ]

    else:

        raise ValueError(
            "'names' must be a list or dictionary."
        )

    # ------------------------------------------------------
    # Validate number of classes
    # ------------------------------------------------------

    if len(class_names) != 2:

        raise ValueError(
            f"Expected exactly 2 classes, "
            f"but found {len(class_names)}."
        )

    # ------------------------------------------------------
    # Validate nc against class names
    # ------------------------------------------------------

    if data.get("nc") != len(class_names):

        raise ValueError(
            f"Mismatch between nc ({data.get('nc')}) "
            f"and number of classes ({len(class_names)})."
        )

    # ------------------------------------------------------
    # Validate class names
    # ------------------------------------------------------

    normalized_names = {
        str(name).strip().lower()
        for name in class_names
    }

    expected_names = {
        "leaf",
        "anthracnose"
    }

    if normalized_names != expected_names:

        raise ValueError(
            "Unexpected class names.\n"
            f"Expected: {expected_names}\n"
            f"Found: {normalized_names}"
        )

    print("\nClass configuration:")

    for index, name in enumerate(class_names):

        print(f"  Class {index}: {name}")

    print(f"\nNumber of classes (nc): {data['nc']}")

    # ------------------------------------------------------
    # Validate train and validation paths
    # ------------------------------------------------------

    required_splits = ["train", "val"]

    resolved_paths = {}

    for split in required_splits:

        if split not in data:

            raise ValueError(
                f"Missing '{split}' entry in data.yaml.\n"
                "Ultralytics requires 'val' as the validation key."
            )

        resolved_path = resolve_dataset_path(
            data[split],
            dataset_root,
            yaml_dir
        )

        if not resolved_path.exists():

            raise FileNotFoundError(
                f"{split.capitalize()} images directory not found:\n"
                f"Configured path: {data[split]}\n"
                f"Resolved path: {resolved_path}"
            )

        if not resolved_path.is_dir():

            raise NotADirectoryError(
                f"{split.capitalize()} path is not a directory:\n"
                f"{resolved_path}"
            )

        resolved_paths[split] = resolved_path

        print(
            f"\n{split.capitalize()} images:\n"
            f"  {resolved_path}"
        )

    # ------------------------------------------------------
    # Resolve test path if available
    # ------------------------------------------------------

    if "test" in data:

        test_path = resolve_dataset_path(
            data["test"],
            dataset_root,
            yaml_dir
        )

        if test_path.exists():

            resolved_paths["test"] = test_path

            print(
                f"\nTest images:\n"
                f"  {test_path}"
            )

        else:

            print(
                "\nWARNING: Test path is configured "
                "but was not found."
            )

            print(f"Configured test path: {data['test']}")

    print("\nDataset configuration validation passed.")

    return data, dataset_root, resolved_paths


# ==========================================================
# Create runtime YAML with resolved paths
# ==========================================================

def create_runtime_yaml(data, dataset_root, resolved_paths):

    runtime_data = data.copy()

    # ------------------------------------------------------
    # Use resolved dataset root
    # ------------------------------------------------------

    runtime_data["path"] = str(dataset_root)

    # ------------------------------------------------------
    # Use absolute paths for dataset splits
    # ------------------------------------------------------

    for split, path in resolved_paths.items():

        runtime_data[split] = str(path)

    # ------------------------------------------------------
    # Save runtime configuration
    # ------------------------------------------------------

    PROJECT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    runtime_yaml = PROJECT_DIR / "runtime_data.yaml"

    with open(
        runtime_yaml,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            runtime_data,
            file,
            sort_keys=False
        )

    print("\nRuntime dataset configuration:")
    print(runtime_yaml)

    return runtime_yaml


# ==========================================================
# Train model
# ==========================================================

def train_model():

    print("=" * 60)
    print("YOLOv8n-Seg TRAINING")
    print("Mango Anthracnose Instance Segmentation")
    print("=" * 60)

    # ------------------------------------------------------
    # Validate dataset configuration
    # ------------------------------------------------------

    data, dataset_root, resolved_paths = validate_dataset()

    # ------------------------------------------------------
    # Create runtime YAML
    # ------------------------------------------------------

    runtime_yaml = create_runtime_yaml(
        data,
        dataset_root,
        resolved_paths
    )

    # ------------------------------------------------------
    # Display training configuration
    # ------------------------------------------------------

    print("\nTraining configuration:")

    print(f"Model       : {MODEL_NAME}")
    print("Task        : Instance segmentation")
    print(f"Classes     : {data['nc']}")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Device      : {DEVICE}")
    print(f"Workers     : {WORKERS}")
    print(f"Seed        : {SEED}")

    # ------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    results = model.train(

        # Dataset configuration
        data=str(runtime_yaml),

        # Training parameters
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Hardware configuration
        device=DEVICE,
        workers=WORKERS,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Validation
        val=True,

        # Saving
        save=True,
        save_period=-1,

        # Output directories
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,

        # Logging and plots
        plots=True,
        verbose=True,
    )

    # ------------------------------------------------------
    # Training summary
    # ------------------------------------------------------

    run_dir = PROJECT_DIR / RUN_NAME

    best_model_path = (
        run_dir / "weights" / "best.pt"
    )

    last_model_path = (
        run_dir / "weights" / "last.pt"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"\nTraining output directory:\n{run_dir}")

    print("\nBest model:")

    if best_model_path.exists():

        print(best_model_path)

    else:

        print("WARNING: best.pt was not found.")

    print("\nLast model:")

    if last_model_path.exists():

        print(last_model_path)

    else:

        print("WARNING: last.pt was not found.")

    print("\nTraining metrics and plots are saved in the run directory.")

    return results


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train_model()
