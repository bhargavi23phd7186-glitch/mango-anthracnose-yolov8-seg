from pathlib import Path
from collections import Counter
import math
import yaml

# ==========================================================
# Dataset organization and Train–Validation–Test verification
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "mango_dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Verify these IDs against data.yaml
LEAF_CLASS_ID = 0
DISEASE_CLASS_ID = 1

EXPECTED_SPLIT_COUNTS = {
    "train": 2100,
    "valid": 450,
    "test": 450,
}


# ==========================================================
# Load and validate class configuration
# ==========================================================

def load_class_names():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Missing dataset configuration file: {DATA_YAML}"
        )

    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names")

    if names is None:
        raise ValueError(
            "The data.yaml file does not contain 'names'."
        )

    # Support list or dictionary class names
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
            "'names' must be a list or dictionary in data.yaml."
        )

    if not class_names:
        raise ValueError("No classes found in data.yaml.")

    print("\nClasses loaded from data.yaml:")

    for class_id, class_name in sorted(class_names.items()):
        print(f"  {class_id}: {class_name}")

    # Validate configured class IDs
    if LEAF_CLASS_ID not in class_names:

        raise ValueError(
            f"LEAF_CLASS_ID={LEAF_CLASS_ID} "
            "is not defined in data.yaml."
        )

    if DISEASE_CLASS_ID not in class_names:

        raise ValueError(
            f"DISEASE_CLASS_ID={DISEASE_CLASS_ID} "
            "is not defined in data.yaml."
        )

    if LEAF_CLASS_ID == DISEASE_CLASS_ID:

        raise ValueError(
            "Leaf and disease class IDs must be different."
        )

    return class_names


# ==========================================================
# Dataset utilities
# ==========================================================

def get_images(image_dir):

    return sorted(
        file for file in image_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_label_files(label_dir):

    return sorted(
        file for file in label_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".txt"
    )


# ==========================================================
# Validate YOLO segmentation annotations
# ==========================================================

def validate_segmentation_line(line, class_names):

    values = line.strip().split()

    if len(values) < 7:

        return False, "Insufficient polygon coordinates"

    try:

        class_id = int(values[0])

        coordinates = [
            float(value)
            for value in values[1:]
        ]

    except ValueError:

        return False, "Invalid numeric values"

    # ------------------------------------------------------
    # Validate class ID
    # ------------------------------------------------------

    if class_id not in class_names:

        return False, f"Unknown class ID: {class_id}"

    # ------------------------------------------------------
    # Validate polygon coordinate structure
    # ------------------------------------------------------

    if len(coordinates) % 2 != 0:

        return False, "Odd number of polygon coordinates"

    if len(coordinates) < 6:

        return False, "Polygon has fewer than 3 points"

    # ------------------------------------------------------
    # Validate finite and normalized coordinates
    # ------------------------------------------------------

    if any(
        not math.isfinite(coordinate)
        or coordinate < 0
        or coordinate > 1
        for coordinate in coordinates
    ):

        return False, (
            "Invalid or out-of-range polygon coordinates"
        )

    return True, class_id


# ==========================================================
# Analyze one dataset split
# ==========================================================

def analyze_split(split_name, class_names):

    split_dir = DATASET_DIR / split_name

    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    print("\n" + "=" * 60)
    print(f"ANALYZING: {split_name.upper()} SET")
    print("=" * 60)

    # ------------------------------------------------------
    # Directory validation
    # ------------------------------------------------------

    if not image_dir.exists():

        print(
            f"ERROR: Missing image directory: {image_dir}"
        )

        return None

    if not label_dir.exists():

        print(
            f"ERROR: Missing label directory: {label_dir}"
        )

        return None

    images = get_images(image_dir)
    labels = get_label_files(label_dir)

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    # ------------------------------------------------------
    # Image-label matching
    # ------------------------------------------------------

    missing_labels = sorted(
        image_stems - label_stems
    )

    missing_images = sorted(
        label_stems - image_stems
    )

    print(f"Images found          : {len(images)}")
    print(f"Label files found     : {len(labels)}")
    print(f"Images without labels : {len(missing_labels)}")
    print(f"Labels without images : {len(missing_images)}")

    # ------------------------------------------------------
    # Annotation analysis
    # ------------------------------------------------------

    class_counts = Counter()

    healthy_images = 0
    diseased_images = 0
    empty_label_images = 0

    invalid_annotations = []

    for image in images:

        label_file = label_dir / f"{image.stem}.txt"

        # Images without matching labels are excluded
        # from image-level classification.
        if not label_file.exists():
            continue

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as file:

                lines = file.readlines()

        except Exception as error:

            invalid_annotations.append(
                (
                    label_file.name,
                    f"File reading error: {error}"
                )
            )

            continue

        # --------------------------------------------------
        # Empty annotation file
        # --------------------------------------------------

        if not lines or not any(line.strip() for line in lines):

            empty_label_images += 1
            continue

        valid_annotation_count = 0

        leaf_present = False
        disease_present = False

        # --------------------------------------------------
        # Validate each annotation
        # --------------------------------------------------

        for line_number, line in enumerate(lines, start=1):

            if not line.strip():
                continue

            valid, result = validate_segmentation_line(
                line,
                class_names
            )

            if not valid:

                invalid_annotations.append(
                    (
                        label_file.name,
                        f"Line {line_number}: {result}"
                    )
                )

                continue

            class_id = result

            class_counts[class_id] += 1
            valid_annotation_count += 1

            if class_id == LEAF_CLASS_ID:
                leaf_present = True

            if class_id == DISEASE_CLASS_ID:
                disease_present = True

        # --------------------------------------------------
        # Image-level classification
        # --------------------------------------------------

        # Only images with at least one valid annotation
        # are considered for image-level classification.

        if valid_annotation_count == 0:
            continue

        # Diseased image:
        # Disease annotation is present.
        if disease_present:

            diseased_images += 1

        # Healthy image:
        # Valid leaf annotation is present, and disease
        # annotation is absent.
        elif leaf_present:

            healthy_images += 1

    # ======================================================
    # Split report
    # ======================================================

    print("\nImage-level distribution:")
    print(f"Healthy images  : {healthy_images}")
    print(f"Diseased images : {diseased_images}")
    print(f"Empty labels    : {empty_label_images}")

    print("\nAnnotation instance counts:")

    if class_counts:

        for class_id, count in sorted(class_counts.items()):

            print(
                f"Class {class_id} "
                f"({class_names[class_id]}): "
                f"{count} polygon annotations"
            )

    else:

        print("No valid annotations found.")

    # ------------------------------------------------------
    # Report missing labels
    # ------------------------------------------------------

    if missing_labels:

        print("\nImages without matching labels:")

        for filename in missing_labels[:10]:
            print(f"  {filename}")

        if len(missing_labels) > 10:
            print("  ...")

    # ------------------------------------------------------
    # Report labels without images
    # ------------------------------------------------------

    if missing_images:

        print("\nLabels without matching images:")

        for filename in missing_images[:10]:
            print(f"  {filename}")

        if len(missing_images) > 10:
            print("  ...")

    # ------------------------------------------------------
    # Report invalid annotations
    # ------------------------------------------------------

    if invalid_annotations:

        print("\nInvalid annotations:")

        for filename, message in invalid_annotations[:10]:

            print(f"  {filename}: {message}")

        if len(invalid_annotations) > 10:
            print("  ...")

    return {
        "split": split_name,
        "images": len(images),
        "labels": len(labels),
        "healthy": healthy_images,
        "diseased": diseased_images,
        "empty_labels": empty_label_images,
        "class_counts": class_counts,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "invalid_annotations": len(invalid_annotations),
    }


# ==========================================================
# Main verification
# ==========================================================

def main():

    print("=" * 60)
    print("MANGO ANTHRACNOSE DATASET VERIFICATION")
    print("=" * 60)

    if not DATASET_DIR.exists():

        print("\nERROR: Dataset directory does not exist.")
        print(f"Expected location:\n{DATASET_DIR}")

        return

    # ------------------------------------------------------
    # Load class configuration
    # ------------------------------------------------------

    try:

        class_names = load_class_names()

    except Exception as error:

        print(f"\nCONFIGURATION ERROR: {error}")
        return

    results = []

    # ------------------------------------------------------
    # Analyze all splits
    # ------------------------------------------------------

    for split in SPLITS:

        result = analyze_split(
            split,
            class_names
        )

        if result is not None:
            results.append(result)

    # ------------------------------------------------------
    # Detect missing or incomplete split folders
    # ------------------------------------------------------

    missing_splits = [
        split
        for split in SPLITS
        if split not in [
            result["split"]
            for result in results
        ]
    ]

    # ======================================================
    # Final summary
    # ======================================================

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)

    total_images = 0
    total_healthy = 0
    total_diseased = 0

    total_missing_labels = 0
    total_missing_images = 0
    total_invalid_annotations = 0

    for result in results:

        total_images += result["images"]
        total_healthy += result["healthy"]
        total_diseased += result["diseased"]

        total_missing_labels += result["missing_labels"]
        total_missing_images += result["missing_images"]
        total_invalid_annotations += result["invalid_annotations"]

        print(
            f"{result['split'].capitalize():12} | "
            f"Images: {result['images']:4} | "
            f"Healthy: {result['healthy']:4} | "
            f"Diseased: {result['diseased']:4}"
        )

    print("-" * 60)

    print(f"Total images             : {total_images}")
    print(f"Healthy images           : {total_healthy}")
    print(f"Diseased images          : {total_diseased}")
    print(f"Missing label files      : {total_missing_labels}")
    print(f"Labels without images    : {total_missing_images}")
    print(f"Invalid annotations      : {total_invalid_annotations}")

    # ------------------------------------------------------
    # Expected split comparison
    # ------------------------------------------------------

    print("\nExpected split comparison:")

    split_count_errors = 0

    for result in results:

        split_name = result["split"]
        actual = result["images"]

        expected = EXPECTED_SPLIT_COUNTS[split_name]

        status = (
            "MATCH"
            if actual == expected
            else "CHECK"
        )

        if actual != expected:
            split_count_errors += 1

        print(
            f"{split_name.capitalize():12} | "
            f"Expected: {expected:4} | "
            f"Actual: {actual:4} | "
            f"{status}"
        )

    # ------------------------------------------------------
    # Missing split report
    # ------------------------------------------------------

    if missing_splits:

        print("\nMissing or incomplete split folders:")

        for split in missing_splits:
            print(f"  - {split}")

    # ======================================================
    # Final status
    # ======================================================

    issues_found = (
        total_missing_labels > 0
        or total_missing_images > 0
        or total_invalid_annotations > 0
        or split_count_errors > 0
        or len(missing_splits) > 0
    )

    print("\n" + "=" * 60)

    if issues_found:

        print("FINAL STATUS: WARNING")
        print("Dataset verification detected issues.")
        print("Review the reports above before using the dataset.")

    else:

        print("FINAL STATUS: PASS")
        print("No structural or annotation validation issues detected.")

    print("=" * 60)

    print("\nVerification completed.")
    print("No images or annotation files were modified.")


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()
