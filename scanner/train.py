"""Training CLI entry point.

Usage:
    python -m scanner.train --model hybrid
    python -m scanner.train --model hybrid --version v1.0
    python -m scanner.train --compare-to v1.0
"""

import argparse
import logging
import sys

from scanner.db import init_db
from scanner.models.training_pipeline import train_model

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
    args = parser.parse_args()

    init_db()

    logger.info("Starting training: model=%s, version=%s", args.model, args.version)

    results = train_model(
        model_type=args.model,
        version=args.version,
        save_path=args.save_path,
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
    print("=" * 60)

    # Compare to previous version if specified
    if args.compare_to:
        print(f"\nComparison with {args.compare_to}:")
        print("  (Comparison functionality to be implemented)")


if __name__ == "__main__":
    main()
