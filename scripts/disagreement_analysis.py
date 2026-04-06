"""Disagreement analysis between auto-labels and human labels.

Analyzes where the automated labeling disagrees with human judgment
to identify improvements for pattern detection, labeling thresholds,
and training data quality.

Usage:
    python scripts/disagreement_analysis.py
    python scripts/disagreement_analysis.py --min-samples 50
    python scripts/disagreement_analysis.py --output report.md
"""

import argparse
import logging
import sys
from typing import Optional

import pandas as pd
import numpy as np

from scanner.labeler import get_human_labeled_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_disagreements(min_samples: int = 10) -> dict:
    """Analyze disagreements between auto-labels and human labels.

    Args:
        min_samples: Minimum number of human-labeled samples required.

    Returns:
        Dict with analysis results.
    """
    # Load human-labeled data
    df = get_human_labeled_data()

    if len(df) < min_samples:
        logger.error(
            "Need at least %d human-labeled samples. Currently have %d. "
            "Review more patterns in the labeling app first.",
            min_samples,
            len(df),
        )
        return {"error": f"Insufficient samples: {len(df)} < {min_samples}"}

    results = {
        "total_samples": len(df),
        "by_strategy": {},
        "by_pattern_type": {},
        "disagreement_types": {},
        "feature_analysis": {},
        "recommendations": [],
    }

    # Analyze each label strategy
    strategies = {
        "outcome_asym_20_7": "Asymmetric 20/7 (+20% gain, -7% loss)",
        "outcome_asym_15_10": "Asymmetric 15/10 (+15% gain, -10% loss)",
        "outcome_sym_10": "Symmetric 10 (+/-10%)",
        "outcome": "Legacy (default)",
    }

    for col, name in strategies.items():
        if col not in df.columns:
            continue

        # Filter to rows where both human and auto labels exist
        valid = df[df[col].isin(["success", "failure"])].copy()
        if len(valid) == 0:
            continue

        # Calculate agreement
        agree = (valid["human_label"] == valid[col]).sum()
        total = len(valid)
        rate = agree / total if total > 0 else 0

        results["by_strategy"][col] = {
            "name": name,
            "agreement_rate": rate,
            "agreed": int(agree),
            "total": total,
            "disagreed": total - agree,
        }

    # Analyze by pattern type
    for pattern_type in df["pattern_type"].unique():
        subset = df[df["pattern_type"] == pattern_type]
        if len(subset) < 3:  # Skip if too few samples
            continue

        agree = (subset["human_label"] == subset["outcome"]).sum()
        total = len(subset)
        rate = agree / total if total > 0 else 0

        results["by_pattern_type"][pattern_type] = {
            "agreement_rate": rate,
            "agreed": int(agree),
            "total": total,
        }

    # Analyze disagreement types
    disagreements = df[df["human_label"] != df["outcome"]].copy()

    if len(disagreements) > 0:
        # Human says success, auto says failure
        h_success_a_fail = disagreements[
            (disagreements["human_label"] == "success")
            & (disagreements["outcome"] == "failure")
        ]
        # Human says failure, auto says success
        h_fail_a_success = disagreements[
            (disagreements["human_label"] == "failure")
            & (disagreements["outcome"] == "success")
        ]

        results["disagreement_types"] = {
            "human_success_auto_failure": {
                "count": len(h_success_a_fail),
                "interpretation": "Auto-labels may be too strict (thresholds too high)",
                "symbols": list(h_success_a_fail["symbol"].head(10)),
            },
            "human_failure_auto_success": {
                "count": len(h_fail_a_success),
                "interpretation": "Auto-labels may be too loose (thresholds too low)",
                "symbols": list(h_fail_a_success["symbol"].head(10)),
            },
        }

    # Analyze features in disagreements
    if len(disagreements) > 5:
        feature_cols = [
            "base_depth_pct",
            "base_duration_weeks",
            "breakout_volume_ratio",
            "rs_line_slope_4wk",
            "quality_score",
        ]

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Compare feature values in agreements vs disagreements
            agreements = df[df["human_label"] == df["outcome"]]

            agree_mean = agreements[col].mean() if len(agreements) > 0 else 0
            disagree_mean = disagreements[col].mean() if len(disagreements) > 0 else 0

            if pd.notna(agree_mean) and pd.notna(disagree_mean):
                diff_pct = (
                    (disagree_mean - agree_mean) / agree_mean * 100
                    if agree_mean != 0
                    else 0
                )

                results["feature_analysis"][col] = {
                    "agree_mean": float(agree_mean),
                    "disagree_mean": float(disagree_mean),
                    "diff_pct": float(diff_pct),
                }

    # Generate recommendations
    results["recommendations"] = _generate_recommendations(results)

    return results


def _generate_recommendations(results: dict) -> list[str]:
    """Generate actionable recommendations based on analysis."""
    recs = []

    # Check pattern type agreement
    for pattern_type, data in results.get("by_pattern_type", {}).items():
        if data["agreement_rate"] < 0.8 and data["total"] >= 5:
            recs.append(
                f"Pattern '{pattern_type}' has {data['agreement_rate']:.0%} agreement "
                f"({data['total']} samples). Review detection criteria in config.yaml."
            )

    # Check disagreement types
    disagree = results.get("disagreement_types", {})
    h_s_a_f = disagree.get("human_success_auto_failure", {}).get("count", 0)
    h_f_a_s = disagree.get("human_failure_auto_success", {}).get("count", 0)

    total_disagree = h_s_a_f + h_f_a_s
    if total_disagree > 0:
        if h_s_a_f > h_f_a_s * 1.5:
            recs.append(
                f"Auto-labels appear too strict: {h_s_a_f} cases where human said "
                f"success but auto said failure. Consider lowering min_gain_pct "
                f"or raising max_loss_pct in config.yaml."
            )
        elif h_f_a_s > h_s_a_f * 1.5:
            recs.append(
                f"Auto-labels appear too loose: {h_f_a_s} cases where human said "
                f"failure but auto said success. Consider raising min_gain_pct "
                f"or lowering max_loss_pct in config.yaml."
            )

    # Check which strategy matches best
    best_strategy = None
    best_rate = 0
    for strategy, data in results.get("by_strategy", {}).items():
        if data["agreement_rate"] > best_rate:
            best_rate = data["agreement_rate"]
            best_strategy = strategy

    if best_strategy and best_strategy != "outcome":
        recs.append(
            f"Label strategy '{best_strategy}' has highest agreement "
            f"({best_rate:.0%}). Consider using this for training."
        )

    # Check feature differences
    for feature, data in results.get("feature_analysis", {}).items():
        if abs(data["diff_pct"]) > 30:
            direction = "higher" if data["diff_pct"] > 0 else "lower"
            recs.append(
                f"Disagreements have {abs(data['diff_pct']):.0f}% {direction} "
                f"'{feature}' on average. This may be a distinguishing factor."
            )

    if not recs:
        recs.append("No significant issues found. Agreement rate looks good.")

    return recs


def format_report(results: dict) -> str:
    """Format analysis results as a readable report."""
    if "error" in results:
        return f"Error: {results['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("DISAGREEMENT ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal human-labeled samples: {results['total_samples']}")

    # Overall agreement by strategy
    lines.append("\n--- Agreement by Label Strategy ---")
    for strategy, data in results.get("by_strategy", {}).items():
        status = "OK" if data["agreement_rate"] >= 0.8 else "REVIEW"
        lines.append(
            f"  {data['name']}: {data['agreement_rate']:.1%} "
            f"({data['agreed']}/{data['total']}) [{status}]"
        )

    # Agreement by pattern type
    lines.append("\n--- Agreement by Pattern Type ---")
    for pattern_type, data in results.get("by_pattern_type", {}).items():
        status = "OK" if data["agreement_rate"] >= 0.8 else "REVIEW"
        lines.append(
            f"  {pattern_type}: {data['agreement_rate']:.1%} "
            f"({data['agreed']}/{data['total']}) [{status}]"
        )

    # Disagreement breakdown
    lines.append("\n--- Disagreement Breakdown ---")
    for dtype, data in results.get("disagreement_types", {}).items():
        lines.append(f"  {dtype.replace('_', ' ').title()}: {data['count']} cases")
        lines.append(f"    Interpretation: {data['interpretation']}")
        if data.get("symbols"):
            lines.append(f"    Examples: {', '.join(data['symbols'][:5])}")

    # Feature analysis
    if results.get("feature_analysis"):
        lines.append("\n--- Feature Differences (Disagree vs Agree) ---")
        for feature, data in results["feature_analysis"].items():
            direction = "+" if data["diff_pct"] > 0 else ""
            lines.append(
                f"  {feature}: {direction}{data['diff_pct']:.1f}% "
                f"(agree: {data['agree_mean']:.2f}, disagree: {data['disagree_mean']:.2f})"
            )

    # Recommendations
    lines.append("\n--- RECOMMENDATIONS ---")
    for i, rec in enumerate(results.get("recommendations", []), 1):
        lines.append(f"  {i}. {rec}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze disagreements between auto and human labels"
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

    # Run analysis
    results = analyze_disagreements(min_samples=args.min_samples)

    # Format report
    report = format_report(results)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

    # Return exit code based on agreement
    if "error" in results:
        sys.exit(1)

    # Check if any strategy has <70% agreement (warning)
    for strategy, data in results.get("by_strategy", {}).items():
        if data["agreement_rate"] < 0.7:
            sys.exit(2)  # Warning: low agreement

    sys.exit(0)


if __name__ == "__main__":
    main()
