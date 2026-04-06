"""Evaluate trained model against human-labeled test set.

This provides an unbiased estimate of model performance by comparing
predictions to human judgment on reviewed patterns.

Usage:
    python scripts/evaluate_against_human.py
    python scripts/evaluate_against_human.py --model models/breakout_lgbm.txt
    python scripts/evaluate_against_human.py --threshold 0.6
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from scanner.labeler import get_human_labeled_data
from scanner.models.data_prep import get_feature_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> lgb.Booster:
    """Load a trained LightGBM model.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded LightGBM Booster.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    return lgb.Booster(model_file=model_path)


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract tabular features from labeled data.

    Args:
        df: DataFrame with pattern data and features.

    Returns:
        Feature array for model prediction.
    """
    feature_names = get_feature_names()
    features = []

    for _, row in df.iterrows():
        row_features = []
        for name in feature_names:
            val = row.get(name, 0)
            if pd.isna(val):
                val = 0
            row_features.append(float(val))
        features.append(row_features)

    return np.array(features, dtype=np.float32)


def evaluate_model(
    model_path: str = "models/breakout_lgbm.txt",
    threshold: float = 0.5,
    min_samples: int = 10,
) -> dict:
    """Evaluate a trained model against human-labeled test set.

    Args:
        model_path: Path to the trained model.
        threshold: Probability threshold for classification.
        min_samples: Minimum number of human-labeled samples required.

    Returns:
        Dict with evaluation metrics and detailed results.
    """
    # Load human-labeled data
    df = get_human_labeled_data()

    if len(df) < min_samples:
        logger.error(
            "Need at least %d human-labeled samples. Currently have %d.",
            min_samples,
            len(df),
        )
        return {"error": f"Insufficient samples: {len(df)} < {min_samples}"}

    logger.info("Loaded %d human-labeled samples for evaluation", len(df))

    # Load model
    try:
        model = load_model(model_path)
        logger.info("Loaded model from %s", model_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return {"error": str(e)}

    # Prepare features
    features = prepare_features(df)
    logger.info("Prepared features: shape %s", features.shape)

    # Get predictions
    probs = model.predict(features)
    preds = (probs >= threshold).astype(int)

    # Ground truth from human labels
    y_true = (df["human_label"] == "success").astype(int).values

    # Calculate metrics
    results = {
        "n_samples": len(df),
        "threshold": threshold,
        "model_path": model_path,
        "metrics": {},
        "confusion_matrix": {},
        "by_pattern_type": {},
        "detailed_predictions": [],
    }

    # Core metrics
    try:
        results["metrics"] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "auc": float(roc_auc_score(y_true, probs)),
        }
    except ValueError as e:
        logger.warning("Could not calculate all metrics: %s", e)
        results["metrics"] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    results["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # Profit factor (how many wins vs losses when model predicts success)
    if tp + fp > 0:
        results["metrics"]["profit_factor"] = tp / max(fp, 1)
        results["metrics"]["win_rate"] = tp / (tp + fp)
    else:
        results["metrics"]["profit_factor"] = 0
        results["metrics"]["win_rate"] = 0

    # Metrics by pattern type
    for pattern_type in df["pattern_type"].unique():
        mask = df["pattern_type"] == pattern_type
        if mask.sum() < 3:
            continue

        subset_true = y_true[mask]
        subset_preds = preds[mask]

        results["by_pattern_type"][pattern_type] = {
            "n_samples": int(mask.sum()),
            "accuracy": float(accuracy_score(subset_true, subset_preds)),
            "precision": float(precision_score(subset_true, subset_preds, zero_division=0)),
            "recall": float(recall_score(subset_true, subset_preds, zero_division=0)),
        }

    # Detailed predictions for review
    for i, (_, row) in enumerate(df.iterrows()):
        results["detailed_predictions"].append({
            "symbol": row["symbol"],
            "pattern_type": row["pattern_type"],
            "pivot_date": row["pivot_date"],
            "human_label": row["human_label"],
            "prob": float(probs[i]),
            "pred": "success" if preds[i] == 1 else "failure",
            "correct": bool(preds[i] == y_true[i]),
        })

    return results


def format_report(results: dict) -> str:
    """Format evaluation results as a readable report."""
    if "error" in results:
        return f"Error: {results['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("MODEL EVALUATION AGAINST HUMAN LABELS")
    lines.append("=" * 60)
    lines.append(f"\nModel: {results['model_path']}")
    lines.append(f"Samples: {results['n_samples']}")
    lines.append(f"Threshold: {results['threshold']}")

    # Core metrics
    lines.append("\n--- Overall Metrics ---")
    m = results["metrics"]
    lines.append(f"  Accuracy:      {m.get('accuracy', 0):.1%}")
    lines.append(f"  Precision:     {m.get('precision', 0):.1%}")
    lines.append(f"  Recall:        {m.get('recall', 0):.1%}")
    lines.append(f"  F1 Score:      {m.get('f1', 0):.1%}")
    if "auc" in m:
        lines.append(f"  AUC:           {m.get('auc', 0):.3f}")
    lines.append(f"  Profit Factor: {m.get('profit_factor', 0):.2f}")
    lines.append(f"  Win Rate:      {m.get('win_rate', 0):.1%}")

    # Confusion matrix
    lines.append("\n--- Confusion Matrix ---")
    cm = results["confusion_matrix"]
    lines.append(f"                 Predicted")
    lines.append(f"                 Failure    Success")
    lines.append(f"  Actual Failure   {cm['true_negatives']:4d}      {cm['false_positives']:4d}")
    lines.append(f"  Actual Success   {cm['false_negatives']:4d}      {cm['true_positives']:4d}")

    # By pattern type
    if results["by_pattern_type"]:
        lines.append("\n--- Metrics by Pattern Type ---")
        for ptype, metrics in results["by_pattern_type"].items():
            lines.append(
                f"  {ptype}: n={metrics['n_samples']}, "
                f"acc={metrics['accuracy']:.1%}, "
                f"prec={metrics['precision']:.1%}, "
                f"rec={metrics['recall']:.1%}"
            )

    # Sample predictions (incorrect ones)
    incorrect = [p for p in results["detailed_predictions"] if not p["correct"]]
    if incorrect:
        lines.append(f"\n--- Incorrect Predictions ({len(incorrect)} total) ---")
        for p in incorrect[:10]:  # Show first 10
            lines.append(
                f"  {p['symbol']} ({p['pattern_type']}): "
                f"human={p['human_label']}, pred={p['pred']} (prob={p['prob']:.2f})"
            )
        if len(incorrect) > 10:
            lines.append(f"  ... and {len(incorrect) - 10} more")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model against human-labeled test set"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/breakout_lgbm.txt",
        help="Path to trained model (default: models/breakout_lgbm.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum human-labeled samples required (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)",
    )
    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        threshold=args.threshold,
        min_samples=args.min_samples,
    )

    # Format report
    report = format_report(results)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    # Return exit code based on accuracy
    if "error" in results:
        sys.exit(1)

    # Warning if accuracy < 70%
    if results["metrics"].get("accuracy", 0) < 0.7:
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
