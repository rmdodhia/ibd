#!/usr/bin/env python3
"""Run labeling strategy experiments to find the best approach.

This script runs 3 experiments with different labeling strategies:
- Experiment A: Original IBD +20%/-7% with focal loss
- Experiment B: Less extreme +15%/-10%
- Experiment C: Symmetric +10%/-10%

Results are compared on AUC, precision, recall, F1, and profit factor.

Usage:
    python scripts/run_label_experiments.py
    python scripts/run_label_experiments.py --model lightgbm
    python scripts/run_label_experiments.py --skip-relabel
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.db import get_connection, init_db
from scanner.labeler import run_labeler, get_labeled_data
from scanner.models.training_pipeline import TrainingPipeline
from scanner.models.lightgbm_trainer import train_lightgbm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_label_stats() -> dict:
    """Get statistics for each label strategy."""
    conn = get_connection()
    try:
        stats = {}

        # Check which columns exist
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(detected_patterns)")
        columns = {row[1] for row in cur.fetchall()}

        strategies = [
            ("asym_20_7", "outcome_asym_20_7"),
            ("asym_15_10", "outcome_asym_15_10"),
            ("sym_10", "outcome_sym_10"),
            ("legacy", "outcome"),
        ]

        for name, col in strategies:
            if col not in columns:
                stats[name] = {"error": f"Column {col} not found"}
                continue

            df = pd.read_sql_query(
                f"""
                SELECT {col} as outcome, COUNT(*) as count
                FROM detected_patterns
                WHERE {col} IS NOT NULL
                GROUP BY {col}
                """,
                conn,
            )

            total = df["count"].sum()
            success = df[df["outcome"] == "success"]["count"].sum() if "success" in df["outcome"].values else 0
            failure = df[df["outcome"] == "failure"]["count"].sum() if "failure" in df["outcome"].values else 0
            neutral = df[df["outcome"] == "neutral"]["count"].sum() if "neutral" in df["outcome"].values else 0
            pending = df[df["outcome"] == "pending"]["count"].sum() if "pending" in df["outcome"].values else 0

            stats[name] = {
                "total": int(total),
                "success": int(success),
                "failure": int(failure),
                "neutral": int(neutral),
                "pending": int(pending),
                "success_rate": round(success / (success + failure) * 100, 1) if (success + failure) > 0 else 0,
                "labeled": int(success + failure),
            }

        return stats
    finally:
        conn.close()


def run_experiment(
    name: str,
    label_strategy: str,
    use_focal_loss: bool,
    model_type: str = "lightgbm",
) -> dict:
    """Run a single experiment.

    Args:
        name: Experiment name for logging.
        label_strategy: Which outcome column to use.
        use_focal_loss: Whether to use focal loss.
        model_type: Model type ('lightgbm' or 'hybrid').

    Returns:
        Dict with experiment results.
    """
    logger.info("=" * 60)
    logger.info("Running Experiment: %s", name)
    logger.info("  Label strategy: %s", label_strategy)
    logger.info("  Focal loss: %s", use_focal_loss)
    logger.info("  Model: %s", model_type)
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        if model_type == "lightgbm":
            # Use LightGBM trainer (faster)
            results = train_lightgbm(label_strategy=label_strategy)
        else:
            # Use hybrid CNN trainer
            pipeline = TrainingPipeline(
                model_type=model_type,
                label_strategy=label_strategy,
                use_focal_loss=use_focal_loss,
            )
            results = pipeline.train(version=f"exp_{label_strategy}")

        duration = (datetime.now() - start_time).total_seconds()

        # Extract key metrics
        metrics = results.get("metrics", {})
        experiment_result = {
            "name": name,
            "label_strategy": label_strategy,
            "use_focal_loss": use_focal_loss,
            "model_type": model_type,
            "auc_mean": metrics.get("auc_mean", 0),
            "auc_std": metrics.get("auc_std", 0),
            "precision_mean": metrics.get("precision_mean", 0),
            "recall_mean": metrics.get("recall_mean", 0),
            "f1_mean": metrics.get("f1_mean", 0),
            "profit_factor_mean": metrics.get("profit_factor_mean", 0),
            "n_samples": metrics.get("total_samples", 0),
            "duration_seconds": round(duration, 1),
            "success": True,
        }

        logger.info("Results for %s:", name)
        logger.info("  AUC: %.3f (+/- %.3f)", experiment_result["auc_mean"], experiment_result["auc_std"])
        logger.info("  Precision: %.3f", experiment_result["precision_mean"])
        logger.info("  Recall: %.3f", experiment_result["recall_mean"])
        logger.info("  F1: %.3f", experiment_result["f1_mean"])
        logger.info("  Profit Factor: %.3f", experiment_result["profit_factor_mean"])

        return experiment_result

    except Exception as e:
        logger.error("Experiment %s failed: %s", name, e)
        return {
            "name": name,
            "label_strategy": label_strategy,
            "use_focal_loss": use_focal_loss,
            "model_type": model_type,
            "error": str(e),
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(description="Run labeling strategy experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "hybrid"],
        help="Model type to use (default: lightgbm for speed)",
    )
    parser.add_argument(
        "--skip-relabel",
        action="store_true",
        help="Skip re-running the labeler (use existing labels)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Which experiments to run (default: A B C)",
    )
    args = parser.parse_args()

    # Initialize database (runs migrations)
    init_db()

    # Step 1: Re-run labeler to populate all label columns
    if not args.skip_relabel:
        logger.info("Step 1: Re-running labeler to populate all label variants...")
        logger.info("This will take a while but computes all 3 strategies at once.")

        # Check if we need to relabel
        stats = get_label_stats()
        needs_relabel = (
            "error" in stats.get("asym_20_7", {})
            or stats.get("asym_20_7", {}).get("labeled", 0) == 0
        )

        if needs_relabel:
            logger.info("New label columns need to be populated. Running labeler with --force...")
            run_labeler(skip_existing=False)
        else:
            logger.info("Label columns already populated. Skipping relabel.")
            logger.info("Use --force with scanner.labeler to recompute if needed.")
    else:
        logger.info("Skipping relabel (--skip-relabel specified)")

    # Step 2: Show label statistics
    logger.info("\n" + "=" * 60)
    logger.info("Label Statistics")
    logger.info("=" * 60)

    stats = get_label_stats()
    for strategy, s in stats.items():
        if "error" in s:
            logger.warning("  %s: %s", strategy, s["error"])
        else:
            logger.info(
                "  %s: %d labeled (%d success, %d failure) = %.1f%% success rate",
                strategy,
                s["labeled"],
                s["success"],
                s["failure"],
                s["success_rate"],
            )

    # Step 3: Define experiments
    experiments = {
        "A": {
            "name": "Exp A: Original IBD +20%/-7% with Focal Loss",
            "label_strategy": "asym_20_7",
            "use_focal_loss": True,
        },
        "B": {
            "name": "Exp B: Less Extreme +15%/-10%",
            "label_strategy": "asym_15_10",
            "use_focal_loss": False,
        },
        "C": {
            "name": "Exp C: Symmetric +10%/-10%",
            "label_strategy": "sym_10",
            "use_focal_loss": False,
        },
    }

    # Step 4: Run experiments
    results = []
    for exp_id in args.experiments:
        if exp_id in experiments:
            exp = experiments[exp_id]
            result = run_experiment(
                name=exp["name"],
                label_strategy=exp["label_strategy"],
                use_focal_loss=exp["use_focal_loss"],
                model_type=args.model,
            )
            results.append(result)

    # Step 5: Compare results
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 60)

    # Create comparison table
    successful = [r for r in results if r.get("success", False)]
    if successful:
        print("\n{:<45} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
            "Experiment", "AUC", "Prec", "Recall", "F1", "ProfitF"
        ))
        print("-" * 95)

        for r in successful:
            print("{:<45} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.3f}".format(
                r["name"][:45],
                r["auc_mean"],
                r["precision_mean"],
                r["recall_mean"],
                r["f1_mean"],
                r["profit_factor_mean"],
            ))

        # Find best by different metrics
        print("\n" + "-" * 60)
        best_auc = max(successful, key=lambda x: x["auc_mean"])
        best_f1 = max(successful, key=lambda x: x["f1_mean"])
        best_pf = max(successful, key=lambda x: x["profit_factor_mean"])

        print(f"Best AUC: {best_auc['name']} ({best_auc['auc_mean']:.3f})")
        print(f"Best F1: {best_f1['name']} ({best_f1['f1_mean']:.3f})")
        print(f"Best Profit Factor: {best_pf['name']} ({best_pf['profit_factor_mean']:.3f})")

    # Save results to experiments.md
    experiments_md = Path(__file__).parent.parent / "experiments.md"
    if experiments_md.exists():
        with open(experiments_md, "a") as f:
            f.write(f"\n\n## Label Strategy Experiment Results ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
            f.write("| Experiment | AUC | Precision | Recall | F1 | Profit Factor |\n")
            f.write("|------------|-----|-----------|--------|-----|---------------|\n")
            for r in successful:
                f.write(f"| {r['name'][:40]} | {r['auc_mean']:.3f} | {r['precision_mean']:.3f} | {r['recall_mean']:.3f} | {r['f1_mean']:.3f} | {r['profit_factor_mean']:.3f} |\n")
        logger.info("Results appended to experiments.md")

    # Save full results to JSON
    results_file = Path(__file__).parent.parent / "data" / "experiment_results.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results saved to %s", results_file)

    return results


if __name__ == "__main__":
    main()
