# FT-ICR DOM Random Forest and SHAP Analysis

This repository contains the source code used to train and interpret random forest models for FT-ICR MS dissolved organic matter (DOM) molecular fate prediction.

The workflow classifies molecular formulas into three fate classes:

- `precursor`
- `product`
- `resistant`

Separate models can be run for middle-aged leachate (ML) and old leachate (OL). The script performs stratified train-test splitting, five-fold stratified grid-search cross-validation, multiclass ROC-AUC evaluation, permutation importance, built-in random forest feature importance and SHAP interpretation.

## Repository Structure

```text
scripts/
  random_forest_dom_fate.py
requirements.txt
README.md
LICENSE
```

## Input Data

The input file should be a CSV table. The response column should be named `Class`; if `Class` is absent, the last column is used as the response variable.

Required feature columns:

```text
C, H, O, N, S, MW, DBE, O/C, H/C, N/C, S/C, AImod, NOSC, DBE-O, DBE/C
```

The `Class` column should contain the molecular fate labels:

```text
precursor
product
resistant
```

The molecular formula tables used in the manuscript are not included in this code repository. They can be supplied separately according to the data availability statement of the manuscript.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+.

## Usage

Run the ML model:

```bash
python scripts/random_forest_dom_fate.py \
  --data-file path/to/ML01.csv \
  --save-dir results/random_forest_resultsML01 \
  --leachate ML
```

Run the OL model:

```bash
python scripts/random_forest_dom_fate.py \
  --data-file path/to/OL01.csv \
  --save-dir results/random_forest_resultsOL01 \
  --leachate OL
```

Windows PowerShell examples:

```powershell
python scripts/random_forest_dom_fate.py `
  --data-file "D:\path\to\ML01.csv" `
  --save-dir "D:\path\to\random_forest_resultsML01" `
  --leachate ML

python scripts/random_forest_dom_fate.py `
  --data-file "D:\path\to\OL01.csv" `
  --save-dir "D:\path\to\random_forest_resultsOL01" `
  --leachate OL
```

## Model Settings

Default settings reproduce the workflow used in the study:

- Train-test split: 70:30
- Split type: stratified random split
- Cross-validation: five-fold stratified cross-validation
- Optimization metric: one-vs-rest multiclass ROC-AUC (`roc_auc_ovr`)
- Random forest trees: `300, 400, 450, 500`
- Minimum samples split: `2, 6, 10`
- Minimum samples leaf: `1, 2, 4, 6`
- Maximum features: `sqrt, log2`
- Splitting criteria: `gini, entropy`
- Class weight: `None, balanced`
- ML maximum depth grid: `20, 25, 30, 40`
- OL maximum depth grid: `15, 20, 25, 30`
- SHAP explanation: TreeExplainer, up to 1,000 test molecules by default

## Main Outputs

The script writes the following files to `--save-dir`:

```text
rf_best_model.joblib
rf_best_grid_search_params.csv
rf_grid_search_cv_results.csv
rf_test_metrics.csv
rf_test_classification_report.csv
rf_test_confusion_matrix.csv
rf_test_confusion_matrix.pdf
rf_test_probabilities.csv
rf_test_roc_curve.pdf
rf_test_roc_auc_by_class.csv
rf_permutation_importance.csv
rf_permutation_importance.pdf
rf_builtin_feature_importance.csv
rf_builtin_feature_importance.pdf
rf_shap_summary_<class>.pdf
rf_shap_importance_<class>.csv
rf_shap_importance_total.csv
rf_shap_importance_total_top20.pdf
```

## Reproducibility Notes

All random states are exposed as command-line arguments. By default, the model and cross-validation random state is `42`, and the train-test split random state is `12`.

To rerun a model using previously selected hyperparameters, place `rf_best_grid_search_params.csv` in the output directory and add:

```bash
--use-existing-best-params
```

To skip SHAP calculation:

```bash
--skip-shap
```

## Suggested Manuscript Sentence

The source code used in this study is available at GitHub: https://github.com/ZhouoA/fticr-dom-random-forest-shap.

