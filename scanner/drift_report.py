"""Model drift monitoring.

Compares live prediction performance to backtest expectations.
Flags when retraining is needed.

Usage:
    python -m scanner.drift_report
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from scanner.config import get
from scanner.db import get_cursor, get_connection, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_drift_report() -> dict:
    """Generate drift monitoring report.

    Compares recent live performance to backtest metrics.

    Returns:
        Dict with drift report data.
    """
    init_db()

    # Get latest model metrics from training
    conn = get_connection()
    try:
        model_runs = pd.read_sql_query(
            """
            SELECT *
            FROM model_runs
            ORDER BY run_date DESC
            LIMIT 1
            """,
            conn,
        )

        # Get resolved predictions from last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        predictions = pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            WHERE resolved_date >= ?
              AND actual_outcome IS NOT NULL
            """,
            conn,
            params=[thirty_days_ago],
        )
    finally:
        conn.close()

    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "backtest_metrics": {},
        "live_metrics": {},
        "drift_detected": False,
        "drift_warnings": [],
        "recommendation": None,
    }

    # Backtest metrics
    if not model_runs.empty:
        model = model_runs.iloc[0]
        report["backtest_metrics"] = {
            "model_version": model.get("model_version", "unknown"),
            "precision": float(model.get("precision_score", 0)),
            "recall": float(model.get("recall_score", 0)),
            "f1": float(model.get("f1_score", 0)),
            "trained_samples": int(model.get("n_train_samples", 0)),
        }

    # Live metrics
    if not predictions.empty:
        # Binary classification: success = 1, failure = 0
        y_true = (predictions["actual_outcome"] == "success").astype(int).values
        y_pred = (predictions["confidence_score"] >= 0.6).astype(int).values

        if len(y_true) > 0:
            report["live_metrics"] = {
                "n_predictions": len(predictions),
                "success_rate": float(y_true.mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }

            # Calculate average return
            returns = predictions["actual_return_pct"].dropna()
            if len(returns) > 0:
                report["live_metrics"]["avg_return"] = float(returns.mean())
                report["live_metrics"]["win_rate"] = float((returns > 0).mean())

    # Check for drift
    drift_threshold = get("retraining.drift_threshold", 0.10)

    if report["backtest_metrics"] and report["live_metrics"]:
        bt = report["backtest_metrics"]
        live = report["live_metrics"]

        # Compare precision
        if bt["precision"] > 0:
            precision_drop = (bt["precision"] - live["precision"]) / bt["precision"]
            if precision_drop > drift_threshold:
                report["drift_detected"] = True
                report["drift_warnings"].append(
                    f"Precision dropped {precision_drop:.1%} (backtest: {bt['precision']:.2f}, live: {live['precision']:.2f})"
                )

        # Compare F1
        if bt["f1"] > 0:
            f1_drop = (bt["f1"] - live["f1"]) / bt["f1"]
            if f1_drop > drift_threshold:
                report["drift_detected"] = True
                report["drift_warnings"].append(
                    f"F1 dropped {f1_drop:.1%} (backtest: {bt['f1']:.2f}, live: {live['f1']:.2f})"
                )

    # Check if enough new outcomes for retraining
    trigger_count = get("retraining.trigger_new_outcomes", 100)
    if report["live_metrics"].get("n_predictions", 0) >= trigger_count:
        report["drift_warnings"].append(
            f"Reached {report['live_metrics']['n_predictions']} new outcomes (trigger: {trigger_count})"
        )

    # Recommendation
    if report["drift_detected"]:
        report["recommendation"] = "RETRAIN: Model drift detected above threshold"
    elif report["live_metrics"].get("n_predictions", 0) >= trigger_count:
        report["recommendation"] = "CONSIDER RETRAIN: New data available"
    elif report["live_metrics"].get("n_predictions", 0) < 20:
        report["recommendation"] = "INSUFFICIENT DATA: Need more resolved predictions"
    else:
        report["recommendation"] = "OK: No drift detected"

    return report


def format_report(report: dict) -> str:
    """Format drift report for display."""
    lines = [
        "",
        "=" * 60,
        "MODEL DRIFT REPORT",
        f"Generated: {report['report_date']}",
        "=" * 60,
        "",
    ]

    # Backtest metrics
    bt = report.get("backtest_metrics", {})
    if bt:
        lines.append("BACKTEST METRICS (from training)")
        lines.append("-" * 40)
        lines.append(f"  Model version: {bt.get('model_version', 'N/A')}")
        lines.append(f"  Precision: {bt.get('precision', 0):.3f}")
        lines.append(f"  Recall:    {bt.get('recall', 0):.3f}")
        lines.append(f"  F1 Score:  {bt.get('f1', 0):.3f}")
        lines.append("")

    # Live metrics
    live = report.get("live_metrics", {})
    if live:
        lines.append("LIVE PERFORMANCE (last 30 days)")
        lines.append("-" * 40)
        lines.append(f"  Predictions: {live.get('n_predictions', 0)}")
        lines.append(f"  Success rate: {live.get('success_rate', 0):.1%}")
        lines.append(f"  Precision: {live.get('precision', 0):.3f}")
        lines.append(f"  Recall:    {live.get('recall', 0):.3f}")
        lines.append(f"  F1 Score:  {live.get('f1', 0):.3f}")
        if "avg_return" in live:
            lines.append(f"  Avg Return: {live['avg_return']:+.1f}%")
        if "win_rate" in live:
            lines.append(f"  Win Rate: {live['win_rate']:.1%}")
        lines.append("")

    # Drift warnings
    if report.get("drift_warnings"):
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for warning in report["drift_warnings"]:
            lines.append(f"  ! {warning}")
        lines.append("")

    # Recommendation
    lines.append("RECOMMENDATION")
    lines.append("-" * 40)
    rec = report.get("recommendation", "N/A")
    if "RETRAIN" in rec:
        lines.append(f"  >>> {rec} <<<")
    else:
        lines.append(f"  {rec}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    report = generate_drift_report()
    print(format_report(report))


if __name__ == "__main__":
    main()
