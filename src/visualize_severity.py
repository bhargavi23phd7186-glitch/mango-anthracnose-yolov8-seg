
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

# ==========================================================
# Publication-Quality Visualization of Mango Anthracnose
# Severity Estimation Results
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# Input and output paths
# ----------------------------------------------------------

INPUT_CSV = (
    ROOT
    / "severity_results"
    / "severity_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "severity_results"
    / "publication_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# Manuscript severity thresholds
# ----------------------------------------------------------

T1 = 10.0
T2 = 25.0

CATEGORIES = [
    "Healthy",
    "Mild",
    "Moderate",
    "Severe",
]

# ----------------------------------------------------------
# Validate input
# ----------------------------------------------------------

if not INPUT_CSV.is_file():
    raise FileNotFoundError(
        f"Severity results CSV not found:\n{INPUT_CSV}"
    )

# ----------------------------------------------------------
# Read severity results
# ----------------------------------------------------------

records = []

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    required_columns = {
        "image_name",
        "disease_severity_percent",
        "severity_category",
    }

    if reader.fieldnames is None:
        raise ValueError(
            "CSV file contains no header."
        )

    missing_columns = (
        required_columns
        - set(reader.fieldnames)
    )

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            f"{missing_columns}"
        )

    for row in reader:

        severity_text = (
            row["disease_severity_percent"]
        )

        category = (
            row["severity_category"]
        )

        # Exclude cases where no leaf was detected
        if (
            severity_text is None
            or severity_text.strip() == ""
            or category == "No leaf detected"
        ):
            continue

        try:

            severity = float(
                severity_text
            )

        except ValueError:

            print(
                "WARNING: Invalid severity value "
                f"for {row['image_name']}: "
                f"{severity_text}"
            )

            continue

        # Validate numerical values
        if not math.isfinite(severity):
            print(
                "WARNING: Non-finite severity "
                f"for {row['image_name']}"
            )
            continue

        if severity < 0 or severity > 100:

            print(
                "WARNING: Severity outside "
                f"0-100% for {row['image_name']}"
            )

            continue

        if category not in CATEGORIES:

            print(
                "WARNING: Unexpected category "
                f"{category} for {row['image_name']}"
            )

            continue

        records.append(
            {
                "image_name": row["image_name"],
                "severity": severity,
                "category": category,
            }
        )

# ----------------------------------------------------------
# Check valid records
# ----------------------------------------------------------

if len(records) == 0:
    raise RuntimeError(
        "No valid severity records found "
        "for visualization."
    )

severity_values = [
    record["severity"]
    for record in records
]

category_counts = {
    category: 0
    for category in CATEGORIES
}

for record in records:

    category_counts[
        record["category"]
    ] += 1

total_valid_images = len(records)

print("=" * 60)
print("SEVERITY VISUALIZATION")
print("=" * 60)

print(
    f"Valid images included: "
    f"{total_valid_images}"
)

print("\nCategory counts:")

for category in CATEGORIES:

    count = category_counts[category]

    percentage = (
        count / total_valid_images
    ) * 100

    print(
        f"{category:<12}: "
        f"{count:>5} "
        f"({percentage:.2f}%)"
    )

# ----------------------------------------------------------
# Calculate statistics
# ----------------------------------------------------------

mean_severity = (
    sum(severity_values)
    / len(severity_values)
)

sorted_values = sorted(
    severity_values
)

n = len(sorted_values)

median_severity = sorted_values[
    n // 2
] if n % 2 == 1 else (
    sorted_values[n // 2 - 1]
    + sorted_values[n // 2]
) / 2

min_severity = min(
    severity_values
)

max_severity = max(
    severity_values
)

# ----------------------------------------------------------
# Save summary statistics
# ----------------------------------------------------------

summary_csv = (
    OUTPUT_DIR
    / "severity_summary.csv"
)

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["Metric", "Value"]
    )

    writer.writerow(
        ["Valid images", total_valid_images]
    )

    writer.writerow(
        ["Mean severity (%)", mean_severity]
    )

    writer.writerow(
        ["Median severity (%)", median_severity]
    )

    writer.writerow(
        ["Minimum severity (%)", min_severity]
    )

    writer.writerow(
        ["Maximum severity (%)", max_severity]
    )

    for category in CATEGORIES:

        writer.writerow(
            [
                f"{category} count",
                category_counts[category],
            ]
        )

        writer.writerow(
            [
                f"{category} percentage",
                (
                    category_counts[category]
                    / total_valid_images
                ) * 100,
            ]
        )

# ==========================================================
# FIGURE 1: Severity Category Distribution
# ==========================================================

counts = [
    category_counts[category]
    for category in CATEGORIES
]

fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    CATEGORIES,
    counts
)

ax.set_xlabel(
    "Disease Severity Category",
    fontsize=12
)

ax.set_ylabel(
    "Number of Images",
    fontsize=12
)

ax.set_title(
    "Distribution of Mango Anthracnose Severity",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

ax.set_axisbelow(True)

for bar, count in zip(
    bars,
    counts
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "severity_category_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR
    / "severity_category_distribution.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================================
# FIGURE 2: Disease Severity Histogram
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    severity_values,
    bins=20,
    edgecolor="black"
)

ax.axvline(
    T1,
    linestyle="--",
    label="Mild threshold (10%)"
)

ax.axvline(
    T2,
    linestyle="--",
    label="Moderate threshold (25%)"
)

ax.set_xlabel(
    "Disease Severity (%)",
    fontsize=12
)

ax.set_ylabel(
    "Number of Images",
    fontsize=12
)

ax.set_title(
    "Disease Severity Distribution",
    fontsize=14
)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "disease_severity_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR
    / "disease_severity_histogram.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================================
# FIGURE 3: Severity Box Plot
# ==========================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ax.boxplot(
    severity_values,
    vert=True,
    showmeans=True
)

ax.set_ylabel(
    "Disease Severity (%)",
    fontsize=12
)

ax.set_title(
    "Statistical Distribution of Disease Severity",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "disease_severity_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR
    / "disease_severity_boxplot.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================================
# FIGURE 4: Category Percentage Distribution
# ==========================================================

category_percentages = [
    (
        category_counts[category]
        / total_valid_images
    ) * 100
    for category in CATEGORIES
]

fig, ax = plt.subplots(
    figsize=(8, 5)
)

bars = ax.bar(
    CATEGORIES,
    category_percentages
)

ax.set_xlabel(
    "Disease Severity Category",
    fontsize=12
)

ax.set_ylabel(
    "Percentage of Images (%)",
    fontsize=12
)

ax.set_title(
    "Percentage Distribution of Severity Categories",
    fontsize=14
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

ax.set_axisbelow(True)

for bar, percentage in zip(
    bars,
    category_percentages
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{percentage:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "severity_category_percentages.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR
    / "severity_category_percentages.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print(
    f"Total valid images: "
    f"{total_valid_images}"
)

print(
    f"Mean severity: "
    f"{mean_severity:.2f}%"
)

print(
    f"Median severity: "
    f"{median_severity:.2f}%"
)

print(
    f"Minimum severity: "
    f"{min_severity:.2f}%"
)

print(
    f"Maximum severity: "
    f"{max_severity:.2f}%"
)

print("\nFigures saved in:")
print(OUTPUT_DIR)

print("\nSummary statistics saved:")
print(summary_csv)
