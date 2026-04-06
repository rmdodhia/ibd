"""
20-Minute Diagnostic Baseline

Tests whether ANY signal exists in the top 5 LightGBM features using logistic regression.
If LogReg can't beat AUC 0.55, the problem is labels/features, not model capacity.

Top 5 features from LightGBM importance:
1. sp500_trend_4wk
2. base_depth_pct
3. market_cap_log
4. rs_rank_percentile
5. breakout_volume_ratio
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from scanner.db import init_db
from scanner.models.data_prep import prepare_cnn_dataset, create_walk_forward_splits, get_feature_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Top 5 features from LightGBM importance
TOP_5_FEATURES = [
    "sp500_trend_4wk",
    "base_depth_pct",
    "market_cap_log",
    "rs_rank_percentile",
    "breakout_volume_ratio",
]

# Extended set for comparison
TOP_10_FEATURES = TOP_5_FEATURES + [
    "price_vs_200dma",
    "institutional_pct",
    "up_down_volume_ratio",
    "price_vs_50dma",
    "rs_line_slope_4wk",
]


def run_logistic_regression_diagnostic():
    """Run logistic regression on top features to establish baseline."""

    init_db()

    # Get training data using existing data prep
    logger.info("Loading training data...")
    price_series, tabular, labels, metadata_df = prepare_cnn_dataset()

    if len(labels) == 0:
        logger.error("No data available")
        return None

    feature_names = get_feature_names()

    logger.info(f"Dataset: {len(labels)} samples, {np.sum(labels)} success ({np.mean(labels)*100:.1f}%)")
    logger.info(f"Features available: {feature_names}")

    # Create DataFrame with features
    df = pd.DataFrame(tabular, columns=feature_names)
    df["success"] = labels
    df["breakout_date"] = metadata_df["breakout_date"].values

    # Check which features are available
    available_top5 = [f for f in TOP_5_FEATURES if f in feature_names]
    available_top10 = [f for f in TOP_10_FEATURES if f in feature_names]

    logger.info(f"Available from top 5: {available_top5}")
    logger.info(f"Available from top 10: {available_top10}")

    if len(available_top5) < 3:
        logger.error("Not enough features available for diagnostic")
        return None

    # Create walk-forward splits (same as main training)
    splits = create_walk_forward_splits(metadata_df)
    logger.info(f"Created {len(splits)} walk-forward splits")

    results = {
        "top5": {"auc": [], "precision": [], "recall": [], "f1": []},
        "top10": {"auc": [], "precision": [], "recall": [], "f1": []},
        "all": {"auc": [], "precision": [], "recall": [], "f1": []},
    }

    for split_idx, split in enumerate(splits):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        y_train = train_df["success"].values
        y_test = test_df["success"].values

        # Skip if no positive examples in test
        if y_test.sum() == 0:
            logger.warning(f"Split {split_idx+1}: No positive examples in test set, skipping")
            continue

        for name, features in [
            ("top5", available_top5),
            ("top10", available_top10),
            ("all", feature_names),
        ]:
            X_train = train_df[features].values
            X_test = test_df[features].values

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train logistic regression
            model = LogisticRegression(
                class_weight="balanced",  # Handle imbalance
                max_iter=1000,
                random_state=42,
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)

            # Metrics
            auc = roc_auc_score(y_test, y_prob)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results[name]["auc"].append(auc)
            results[name]["precision"].append(precision)
            results[name]["recall"].append(recall)
            results[name]["f1"].append(f1)

        logger.info(
            f"Split {split_idx+1}/{len(splits)}: "
            f"Top5 AUC={results['top5']['auc'][-1]:.3f}, "
            f"Top10 AUC={results['top10']['auc'][-1]:.3f}, "
            f"All AUC={results['all']['auc'][-1]:.3f}"
        )

    # Print summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC BASELINE RESULTS")
    print("=" * 70)
    print("\nLogistic Regression on walk-forward splits:\n")

    for name, label in [
        ("top5", f"Top 5 features ({len(available_top5)} available)"),
        ("top10", f"Top 10 features ({len(available_top10)} available)"),
        ("all", f"All features ({len(feature_names)} total)"),
    ]:
        if len(results[name]["auc"]) == 0:
            continue

        auc_mean = np.mean(results[name]["auc"])
        auc_std = np.std(results[name]["auc"])
        prec_mean = np.mean(results[name]["precision"])
        recall_mean = np.mean(results[name]["recall"])
        f1_mean = np.mean(results[name]["f1"])

        print(f"{label}:")
        print(f"  AUC:       {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"  Precision: {prec_mean:.3f}")
        print(f"  Recall:    {recall_mean:.3f}")
        print(f"  F1:        {f1_mean:.3f}")
        print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_auc = max(
        np.mean(results["top5"]["auc"]) if results["top5"]["auc"] else 0,
        np.mean(results["top10"]["auc"]) if results["top10"]["auc"] else 0,
        np.mean(results["all"]["auc"]) if results["all"]["auc"] else 0,
    )

    if best_auc < 0.52:
        print("\nRESULT: NO SIGNAL DETECTED")
        print("  AUC < 0.52 indicates features are essentially random.")
        print("  → Problem is definitely labels or features, not model capacity.")
        print("  → Fix labels (symmetric thresholds) before any other work.")

    elif best_auc < 0.55:
        print("\nRESULT: VERY WEAK SIGNAL")
        print("  AUC 0.52-0.55 indicates marginal signal at best.")
        print("  → Complex models (CNN, ResNet) unlikely to help.")
        print("  → Focus on label redesign and new features.")

    elif best_auc < 0.58:
        print("\nRESULT: WEAK BUT PRESENT SIGNAL")
        print("  AUC 0.55-0.58 indicates some learnable signal exists.")
        print("  → Worth trying better features and ensemble methods.")
        print("  → Architecture changes may help moderately.")

    elif best_auc < 0.62:
        print("\nRESULT: MODERATE SIGNAL")
        print("  AUC 0.58-0.62 indicates meaningful signal.")
        print("  → Model improvements likely to help.")
        print("  → Proceed with architecture experiments.")

    else:
        print("\nRESULT: GOOD SIGNAL")
        print("  AUC > 0.62 with simple LogReg is promising.")
        print("  → More complex models should improve further.")
        print("  → Proceed confidently with experimentation.")

    print()

    # Feature coefficients from final model (all data)
    print("=" * 70)
    print("FEATURE COEFFICIENTS (Top 5, trained on all data)")
    print("=" * 70)

    X_all = df[available_top5].values
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_all = df["success"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_scaled, y_all)

    print("\nCoefficients (positive = increases success probability):\n")
    for feat, coef in sorted(zip(available_top5, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = "+" if coef > 0 else "-"
        print(f"  {feat}: {direction}{abs(coef):.3f}")

    print()
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_logistic_regression_diagnostic()
