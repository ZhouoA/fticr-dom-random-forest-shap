# -*- coding: utf-8 -*-
"""Random forest and SHAP workflow for FT-ICR MS DOM molecular fate prediction.

The script trains a multiclass random forest classifier to distinguish DOM
molecules assigned as precursor, product or resistant. It reproduces the
workflow used for the ML01 and OL01 datasets while avoiding hard-coded local
paths so that the analysis can be rerun from the command line.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

warnings.filterwarnings("ignore")


FEATURE_COLUMNS = [
    "C",
    "H",
    "O",
    "N",
    "S",
    "MW",
    "DBE",
    "O/C",
    "H/C",
    "N/C",
    "S/C",
    "AImod",
    "NOSC",
    "DBE-O",
    "DBE/C",
]


def safe_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>| ]+", "_", str(name).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and interpret random forest models for DOM molecular fate prediction."
    )
    parser.add_argument("--data-file", required=True, help="Input CSV file containing feature columns and Class.")
    parser.add_argument("--save-dir", required=True, help="Directory for model outputs.")
    parser.add_argument(
        "--leachate",
        required=True,
        choices=["ML", "OL"],
        help="Leachate type. Used to select the depth grid used in the study.",
    )
    parser.add_argument("--prefix", default="rf", help="Prefix for output files.")
    parser.add_argument("--test-size", type=float, default=0.30, help="Fraction held out as the test set.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for model/CV/permutation/SHAP.")
    parser.add_argument("--split-random-state", type=int, default=12, help="Random state for train-test split.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--scoring", default="roc_auc_ovr", help="Grid-search and permutation scoring metric.")
    parser.add_argument("--grid-n-jobs", type=int, default=-1, help="Parallel workers for grid search.")
    parser.add_argument("--perm-n-repeats", type=int, default=10, help="Permutation importance repeats.")
    parser.add_argument("--shap-sample-n", type=int, default=1000, help="Maximum test molecules explained by SHAP.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP calculation.")
    parser.add_argument(
        "--use-existing-best-params",
        action="store_true",
        help="Load <prefix>_best_grid_search_params.csv from save-dir and skip grid search.",
    )
    return parser.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def load_data(path: Path, cv_splits: int) -> tuple[pd.DataFrame, pd.Series]:
    df = read_csv_robust(path)
    df.columns = df.columns.astype(str).str.strip()

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    if target_col != "Class":
        df = df.rename(columns={target_col: "Class"})

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y_text = df["Class"].astype(str).str.strip()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & y_text.notna() & (y_text != "") & (y_text.str.lower() != "nan")
    X = X.loc[valid].reset_index(drop=True)
    y_text = y_text.loc[valid].reset_index(drop=True)

    counts = y_text.value_counts()
    print("Data file:", path)
    print("Cleaned sample size:", len(X))
    print("Feature count:", X.shape[1])
    print("Feature columns:", X.columns.tolist())
    print("\nClass distribution:")
    print(counts)

    if len(counts) < 2:
        raise ValueError("At least two classes are required for classification.")
    if (counts < cv_splits).any():
        raise ValueError(f"Each class needs at least {cv_splits} samples for {cv_splits}-fold CV.")

    return X, y_text


def build_estimator_and_grid(leachate: str, random_state: int) -> tuple[RandomForestClassifier, dict]:
    depth_grid = [20, 25, 30, 40] if leachate == "ML" else [15, 20, 25, 30]
    estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 400, 450, 500],
        "max_depth": depth_grid,
        "min_samples_split": [2, 6, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
    }
    return estimator, param_grid


def coerce_param_value(key: str, value):
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    text = str(value).strip()
    if text in {"True", "False"}:
        return text == "True"
    if key in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}:
        try:
            return int(float(text))
        except ValueError:
            return None if text.lower() == "none" else text
    return text


def load_existing_best_params(save_dir: Path, prefix: str) -> dict:
    best_file = save_dir / f"{prefix}_best_grid_search_params.csv"
    if not best_file.exists():
        raise FileNotFoundError(f"Best-parameter file not found: {best_file}")
    row = pd.read_csv(best_file).iloc[0].to_dict()
    params = {
        key: coerce_param_value(key, value)
        for key, value in row.items()
        if key not in {"best_cv_auc_ovr", "best_cv_score", "scoring"}
    }
    print("\nLoaded existing best parameters:")
    print(params)
    return params


def fit_model(
    estimator: RandomForestClassifier,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> RandomForestClassifier:
    save_dir = Path(args.save_dir)
    if args.use_existing_best_params:
        model = estimator.set_params(**load_existing_best_params(save_dir, args.prefix))
        model.fit(X_train, y_train)
        print("\nGrid search skipped; refit model using existing best parameters.")
        return model

    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.grid_n_jobs,
        verbose=2,
        refit=True,
        return_train_score=False,
        pre_dispatch="2*n_jobs",
    )
    search.fit(X_train, y_train)

    print("\nBest parameters selected by 5-fold grid search:")
    print(search.best_params_)
    print(f"Best cross-validation score ({args.scoring}):", search.best_score_)

    pd.DataFrame(search.cv_results_).to_csv(
        save_dir / f"{args.prefix}_grid_search_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([search.best_params_]).assign(
        best_cv_auc_ovr=search.best_score_,
        scoring=args.scoring,
    ).to_csv(
        save_dir / f"{args.prefix}_best_grid_search_params.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return search.best_estimator_


def save_test_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    prefix: str,
) -> None:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc_ovr_macro": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "auc_ovr_weighted": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
    }
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}).to_csv(
        save_dir / f"{prefix}_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        )
    ).transpose().to_csv(save_dir / f"{prefix}_test_classification_report.csv", encoding="utf-8-sig")


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    prefix: str,
) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        save_dir / f"{prefix}_test_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_test_confusion_matrix.pdf", format="pdf")
    plt.close(fig)


def save_probabilities(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    prefix: str,
) -> None:
    prob_df = pd.DataFrame(y_prob, columns=[f"prob_{name}" for name in class_names])
    prob_df.insert(0, "y_pred", [class_names[i] for i in y_pred])
    prob_df.insert(0, "y_true", [class_names[i] for i in y_test])
    prob_df.to_csv(save_dir / f"{prefix}_test_probabilities.csv", index=False, encoding="utf-8-sig")


def save_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_dir: Path,
    prefix: str,
) -> None:
    y_binary = label_binarize(y_test, classes=np.arange(len(class_names)))
    rows = []
    plt.figure(figsize=(7, 7))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_binary[:, idx], y_prob[:, idx])
        class_auc = auc(fpr, tpr)
        rows.append({"Class": class_name, "AUC": class_auc})
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={class_auc:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_binary.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    rows.append({"Class": "micro-average", "AUC": auc_micro})
    plt.plot(fpr_micro, tpr_micro, lw=2, linestyle="--", color="black", label=f"micro-average (AUC={auc_micro:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlim([-0.01, 1.02])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Random Forest ROC Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_test_roc_curve.pdf", dpi=600)
    plt.close()

    pd.DataFrame(rows).to_csv(save_dir / f"{prefix}_test_roc_auc_by_class.csv", index=False, encoding="utf-8-sig")


def save_permutation_importance_outputs(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    save_dir = Path(args.save_dir)
    print("\nCalculating permutation importance...")
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=args.perm_n_repeats,
        random_state=args.random_state,
        scoring=args.scoring,
        n_jobs=args.grid_n_jobs,
    )
    perm_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "PermutationImportanceMean": perm.importances_mean,
            "PermutationImportanceStd": perm.importances_std,
        }
    ).sort_values("PermutationImportanceMean", ascending=False)
    perm_df.to_csv(save_dir / f"{args.prefix}_permutation_importance.csv", index=False, encoding="utf-8-sig")

    plot_df = perm_df.head(min(20, len(perm_df))).sort_values("PermutationImportanceMean", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["Feature"], plot_df["PermutationImportanceMean"])
    plt.xlabel("Permutation importance")
    plt.title("Random Forest Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(save_dir / f"{args.prefix}_permutation_importance.pdf", format="pdf")
    plt.close()
    return perm_df


def save_builtin_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    save_dir: Path,
    prefix: str,
) -> None:
    importance = np.asarray(model.feature_importances_, dtype=float)
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance, "Source": "built_in"}).sort_values(
        "Importance", ascending=False
    )
    imp_df.to_csv(save_dir / f"{prefix}_builtin_feature_importance.csv", index=False, encoding="utf-8-sig")

    plot_df = imp_df.head(min(20, len(imp_df))).sort_values("Importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["Feature"], plot_df["Importance"])
    plt.xlabel("Feature importance")
    plt.title("Random Forest Built-in Feature Importance")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_builtin_feature_importance.pdf", format="pdf")
    plt.close()


def normalize_shap_values(shap_values, n_classes: int) -> list[np.ndarray]:
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[2] == n_classes:
            return [arr[:, :, i] for i in range(n_classes)]
        if arr.shape[0] == n_classes:
            return [arr[i, :, :] for i in range(n_classes)]
    raise ValueError(f"Unsupported SHAP values format: {type(shap_values)} with shape {getattr(shap_values, 'shape', None)}")


def save_shap_outputs(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    if args.skip_shap:
        print("\nSkip SHAP outputs because --skip-shap was used.")
        return

    save_dir = Path(args.save_dir)
    print("\nPreparing SHAP analysis...")
    X_shap = X_test.sample(min(args.shap_sample_n, len(X_test)), random_state=args.random_state)
    print("SHAP explained sample size:", len(X_shap))

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_shap)
    shap_by_class = normalize_shap_values(shap_raw, len(class_names))
    feature_names = X_train.columns.tolist()
    shap_mean_list = []

    for idx, class_name in enumerate(class_names):
        if idx >= len(shap_by_class):
            continue
        shap_arr = np.asarray(shap_by_class[idx])
        if shap_arr.ndim != 2:
            shap_arr = np.squeeze(shap_arr)
        if shap_arr.shape[1] != X_shap.shape[1]:
            raise ValueError(
                f"SHAP feature number mismatch for {class_name}: {shap_arr.shape[1]} vs {X_shap.shape[1]}"
            )

        class_safe = safe_filename(class_name)
        plt.figure()
        shap.summary_plot(shap_arr, X_shap, show=False, max_display=min(20, X_shap.shape[1]))
        plt.title(f"Random Forest SHAP Summary - {class_name}")
        plt.tight_layout()
        plt.savefig(save_dir / f"{args.prefix}_shap_summary_{class_safe}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        class_imp = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": np.abs(shap_arr).mean(axis=0)}).sort_values(
            "MeanAbsSHAP", ascending=False
        )
        class_imp.to_csv(save_dir / f"{args.prefix}_shap_importance_{class_safe}.csv", index=False, encoding="utf-8-sig")
        shap_mean_list.append(np.abs(shap_arr).mean(axis=0))

    save_total_shap_importance(feature_names, class_names, shap_mean_list, save_dir, args.prefix)


def save_total_shap_importance(
    feature_names: list[str],
    class_names: list[str],
    shap_mean_list: list[np.ndarray],
    save_dir: Path,
    prefix: str,
) -> None:
    if not shap_mean_list:
        print("No SHAP values were generated.")
        return

    shap_mean = np.stack(shap_mean_list, axis=1)
    total_df = pd.DataFrame({"Feature": feature_names})
    for idx, class_name in enumerate(class_names[: shap_mean.shape[1]]):
        total_df[f"SHAP_{class_name}"] = shap_mean[:, idx]
    total_df["Total"] = total_df.drop(columns=["Feature"]).sum(axis=1)
    total_df = total_df.sort_values("Total", ascending=False)
    total_df.to_csv(save_dir / f"{prefix}_shap_importance_total.csv", index=False, encoding="utf-8-sig")

    top_df = total_df.head(min(20, len(total_df))).sort_values("Total", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top_df["Feature"], top_df["Total"])
    plt.xlabel("Total mean |SHAP value|")
    plt.title("Random Forest Total SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_shap_importance_total_top20.pdf", format="pdf")
    plt.close()
    print("SHAP outputs saved.")


def main() -> None:
    args = parse_args()
    data_file = Path(args.data_file)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y_text = load_data(data_file, args.cv_splits)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = label_encoder.classes_.tolist()
    print("\nClass order:", class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.split_random_state,
        stratify=y,
    )
    print("\nTraining set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    estimator, param_grid = build_estimator_and_grid(args.leachate, args.random_state)
    model = fit_model(estimator, param_grid, X_train, y_train, args)
    joblib.dump(model, save_dir / f"{args.prefix}_best_model.joblib")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    save_test_metrics(y_test, y_pred, y_prob, class_names, save_dir, args.prefix)
    save_confusion_matrix(y_test, y_pred, class_names, save_dir, args.prefix)
    save_probabilities(y_test, y_pred, y_prob, class_names, save_dir, args.prefix)
    save_roc_curve(y_test, y_prob, class_names, save_dir, args.prefix)
    save_permutation_importance_outputs(model, X_test, y_test, args)
    save_builtin_feature_importance(model, X.columns.tolist(), save_dir, args.prefix)
    save_shap_outputs(model, X_train, X_test, class_names, args)

    print("\nAll results saved to:")
    print(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
