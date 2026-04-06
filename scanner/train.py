"""Training CLI entry point.

Usage:
    python -m scanner.train --model hybrid
    python -m scanner.train --model hybrid --version v1.0
    python -m scanner.train --model lightgbm --label-strategy sym_10
    python -m scanner.train --model hybrid --focal-loss
    python -m scanner.train --compare-to v1.0
"""

import argparse
import logging
import sys

from scanner.db import init_db
from scanner.models.training_pipeline import train_model
from scanner.models.lightgbm_trainer import train_lightgbm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train breakout prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["hybrid", "cnn", "lightgbm"],
        help="Model type to train (default: hybrid)",
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version string (default: auto-generated)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save model (default: models/breakout_predictor.pt)",
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        help="Version to compare results against",
    )
    parser.add_argument(
        "--label-strategy",
        type=str,
        choices=["asym_20_7", "asym_15_10", "sym_10"],
        help="Which outcome column to use for labels (default: config setting)",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use focal loss instead of BCE (helps with imbalanced data)",
    )
    args = parser.parse_args()

    init_db()

    logger.info(
        "Starting training: model=%s, version=%s, label_strategy=%s, focal_loss=%s",
        args.model, args.version, args.label_strategy, args.focal_loss
    )

    if args.model == "lightgbm":
        results = train_lightgbm(
            version=args.version,
            save_path=args.save_path,
            label_strategy=args.label_strategy,
        )
    else:
        results = train_model(
            model_type=args.model,
            version=args.version,
            save_path=args.save_path,
            label_strategy=args.label_strategy,
            use_focal_loss=args.focal_loss if args.focal_loss else None,
        )

    if "error" in results:
        logger.error("Training failed: %s", results["error"])
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Version: {results['version']}")
    print(f"Model saved to: {results['model_path']}")
    print()
    print("Aggregate Metrics (across walk-forward splits):")
    metrics = results.get("metrics", {})
    print(f"  Precision: {metrics.get('precision_mean', 0):.3f} ± {metrics.get('precision_std', 0):.3f}")
    print(f"  Recall:    {metrics.get('recall_mean', 0):.3f} ± {metrics.get('recall_std', 0):.3f}")
    print(f"  F1 Score:  {metrics.get('f1_mean', 0):.3f} ± {metrics.get('f1_std', 0):.3f}")
    print(f"  AUC:       {metrics.get('auc_mean', 0):.3f} ± {metrics.get('auc_std', 0):.3f}")
    print(f"  Profit Factor: {metrics.get('profit_factor_mean', 0):.2f}")
    print()
    print(f"Evaluated on {metrics.get('n_splits', 0)} splits, {metrics.get('total_samples', 0)} total samples")

    # Print feature importance for LightGBM
    if "feature_importance" in results:
        print()
        print("Top 10 Feature Importance:")
        for item in results["feature_importance"][:10]:
            print(f"  {item['feature']}: {item['importance']:.2f}")

    print("=" * 60)

    # Compare to previous version if specified
    if args.compare_to:
        print(f"\nComparison with {args.compare_to}:")
        print("  (Comparison functionality to be implemented)")


if __name__ == "__main__":
    main()
