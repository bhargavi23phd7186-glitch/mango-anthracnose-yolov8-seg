# Results

## 1. Detection and Segmentation Evaluation

The trained YOLOv8-Seg model is evaluated using the independent test dataset to assess mango leaf detection and anthracnose lesion segmentation performance.

The evaluation includes the following metrics:

- Precision
- Recall
- F1-score
- Box mAP@0.5
- Box mAP@0.5–0.95
- Mask mAP@0.5
- Mask mAP@0.5–0.95

The evaluation results are generated using the evaluation script provided in the `src/` directory.

## 2. Image-Level Disease Classification

Image-level classification evaluation is performed to distinguish healthy mango leaves from diseased leaves.

The classification evaluation includes:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Confusion matrix
- Receiver Operating Characteristic (ROC) curve
- Area Under the Curve (AUC)

The confusion matrix and ROC curve are generated using the independent test dataset.

## 3. Disease Severity Estimation

Disease severity is estimated from the predicted leaf and anthracnose segmentation masks.

The disease severity percentage is calculated as:

\[
\text{Disease Severity (\%)} =
\frac{\text{Anthracnose Area}}{\text{Leaf Area}}
\times 100
\]

The estimated severity percentage is used to categorize disease severity.

## 4. Severity Classification

Disease severity is classified into the following categories:

| Category | Severity Percentage |
|---|---|
| Healthy | 0% |
| Mild | >0–10% |
| Moderate | >10–25% |
| Severe | >25% |

## 5. Visualization and Statistical Analysis

The project generates visualizations and statistical summaries of disease severity, including:

- Disease severity distribution
- Severity category distribution
- Severity histogram
- Box plot
- Category percentage visualization

These figures are generated using the severity visualization script.

## 6. Generated Outputs

The evaluation scripts generate the following outputs:

- Evaluation metrics
- CSV files containing individual image predictions
- JSON files containing evaluation metrics
- Classification reports
- Confusion matrix
- ROC curve
- Disease severity results
- Severity distribution figures

Generated outputs are saved locally in the project results directory.

## 7. Reproducibility

The Python scripts provided in the `src/` directory can be used to reproduce the model training, evaluation, image-level classification, disease severity estimation, and visualization workflows.

Reproduction requires the dataset, trained model weights where applicable, and the required Python dependencies.
