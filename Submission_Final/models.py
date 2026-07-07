"""Stage 3 model comparison: XGBoost (primary) vs. MLP neural baseline.

Exports:
    load_data        — load MFCC feature table, return train/test splits
    build_xgboost    — configure the primary XGBoost classifier
    build_mlp        — configure the MLP neural baseline
    evaluate_model   — return accuracy, macro AUC, predictions, probabilities
    plot_confusion_matrix  — save confusion matrix figure to figures/
    plot_metric_comparison — save side-by-side bar chart to figures/
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

FEATURE_COLS: list[str] = (
    [f"mfcc_{i:02d}_mean" for i in range(1, 14)]
    + [f"mfcc_{i:02d}_std" for i in range(1, 14)]
)
SEED = 42
FIGURES_DIR = Path("figures")


def load_data(
    features_path: str | Path = "data/processed/mfcc_features.csv",
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    LabelEncoder, pd.DataFrame, pd.DataFrame
]:
    """Load MFCC features; combine train+val for fitting, hold test for evaluation.

    Combining train+val (15 samples) maximises the data available for fitting
    while still providing a held-out test set (5 samples) that the model
    never sees during training.
    """
    df = pd.read_csv(features_path)
    train_df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"])
    y_test = le.transform(test_df["label"])

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, le, train_df, test_df


def build_xgboost() -> xgb.XGBClassifier:
    """Shallow gradient-boosted trees — trains in seconds on a laptop CPU.

    Hyperparameter choices:
        n_estimators=50   — enough for 5-class data; avoids overfitting on n=15
        max_depth=2       — keeps individual trees shallow (stump + one split)
        learning_rate=0.3 — standard starting point for small datasets
    """
    return xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.3,
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )


def build_mlp() -> Pipeline:
    """Two-layer MLP as the neural comparison baseline.

    The original proposal described a 1D CNN.  Since PyTorch/TensorFlow are
    outside the project stack, MLPClassifier is the appropriate substitute:
    on 26-dimensional summary features (MFCC mean + std) the CNN's temporal
    inductive bias over the raw time axis is already discarded by the pooling
    step, so MLP and a shallow 1D CNN have equivalent expressive power here.

    StandardScaler is prepended because MLP gradient descent is sensitive to
    feature scale, unlike XGBoost which is scale-invariant.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=SEED,
        )),
    ])


def build_dummy(strategy: str = "most_frequent") -> DummyClassifier:
    """Naive reference baseline — not a real candidate model, just a sanity-check floor.

    strategy="most_frequent" always predicts the majority training class.
    strategy="stratified" samples predictions according to the training class
    distribution. Both ignore the input features entirely, so any real model
    that fails to beat this baseline is not learning anything from the audio.
    """
    return DummyClassifier(strategy=strategy, random_state=SEED)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    model_name: str,
) -> dict:
    """Return accuracy, macro OvR AUC, per-class report, predictions, and probabilities."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "model": model_name,
        "accuracy": report["accuracy"],
        "macro_auc": auc,
        "report": report,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_name: str,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", pad=10)
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_").replace("-", "_")
    out = figures_dir / f"confusion_{slug}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def run_seed_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    n_seeds: int = 5,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Retrain XGBoost and MLP across multiple seeds and summarise stability.

    seed=42 remains the "primary" reproducible run used everywhere else in the
    notebook. This sweep is an additional robustness check on top of that run,
    not a replacement — with n=15-30 training samples, a single seed's metrics
    can easily be an outlier rather than a representative estimate.
    """
    if seeds is None:
        seeds = list(range(n_seeds))

    records: list[dict] = []
    for seed in seeds:
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.3,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_res = evaluate_model(xgb_model, X_test, y_test, label_names, "XGBoost")

        mlp_model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=seed,
            )),
        ])
        mlp_model.fit(X_train, y_train)
        mlp_res = evaluate_model(mlp_model, X_test, y_test, label_names, "MLP Baseline")

        for res in (xgb_res, mlp_res):
            records.append({
                "seed": seed,
                "model": res["model"],
                "accuracy": res["accuracy"],
                "macro_auc": res["macro_auc"],
            })

    sweep_df = pd.DataFrame(records)
    summary = sweep_df.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        macro_auc_mean=("macro_auc", "mean"),
        macro_auc_std=("macro_auc", "std"),
        macro_auc_min=("macro_auc", "min"),
        macro_auc_max=("macro_auc", "max"),
    ).reset_index()
    return summary


def plot_metric_comparison(
    results: list[dict],
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    names = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    aucs = [
        r["macro_auc"] if not np.isnan(r["macro_auc"]) else 0.0
        for r in results
    ]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, aucs, width, label="Macro AUC (OvR)", color="#e76f51")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_title("Model Comparison — Accuracy vs. Macro AUC", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out = figures_dir / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
