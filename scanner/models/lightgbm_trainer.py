"""LightGBM training module for breakout prediction.

This provides a tree-based baseline using only tabular features.
"""

import logging
from datetime import datetime
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
)

from scanner.config import get
from scanner.db import get_connection
from scanner.models.data_prep import (
    prepare_cnn_dataset,
    create_walk_forward_splits,
    get_feature_names,
)

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """Train LightGBM model for breakout prediction."""

    def __init__(self, label_strategy: str = None):
        """Initialize trainer with config parameters.

        Args:
            label_strategy: Which outcome column to use ('asym_20_7', 'asym_15_10', 'sym_10').
        """
        self.label_strategy = label_strategy or get("training.label_strategy", "asym_20_7")
        self.num_leaves_range = get("training.lightgbm.num_leaves_range", [20, 150])
        self.learning_rate_range = get("training.lightgbm.learning_rate_range", [0.01, 0.3])
        self.min_child_samples_range = get("training.lightgbm.min_child_samples_range", [5, 50])

        logger.info("Label strategy: %s", self.label_strategy)

        # Fixed parameters for initial training
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 50,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": get("training.random_seed", 42),
        }

    def train(
        self,
        version: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        """Train LightGBM model with walk-forward validation.

        Args:
            version: Model version string.
            save_path: Path to save model.

        Returns:
            Training results dict with metrics and feature importance.
        """
        if version is None:
            version = f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("Starting LightGBM training for version %s", version)
        logger.info("Label strategy: %s", self.label_strategy)

        # Prepare dataset - we only need tabular features for LightGBM
        price_series, tabular, labels, metadata_df = prepare_cnn_dataset(
            label_strategy=self.label_strategy
        )

        logger.info("Dataset: %d samples, %d success (%.1f%%)",
                   len(labels), labels.sum(), 100 * labels.mean())

        # Create walk-forward splits
        splits = create_walk_forward_splits(metadata_df)
        logger.info("Created %d walk-forward splits", len(splits))

        all_metrics = []
        all_predictions = []
        feature_importance_sum = np.zeros(tabular.shape[1])

        feature_names = get_feature_names()

        for i, split in enumerate(splits):
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            X_train = tabular[train_idx]
            y_train = labels[train_idx]
            X_test = tabular[test_idx]
            y_test = labels[test_idx]

            logger.info(
                "Split %d/%d: Train %d samples, Test %d samples (%.1f%% success)",
                i + 1, len(splits), len(train_idx), len(test_idx),
                100 * y_test.mean()
            )

            # Handle class imbalance
            pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            params = self.params.copy()
            params["scale_pos_weight"] = pos_weight

            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
            val_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=train_data)

            # Train with early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100),
                ],
            )

            # Evaluate
            probs = model.predict(X_test)
            preds = (probs >= 0.5).astype(int)

            # Calculate profit factor (wins / losses on predictions)
            wins = np.sum((preds == 1) & (y_test == 1))
            losses = np.sum((preds == 1) & (y_test == 0))
            profit_factor = wins / max(losses, 1)

            metrics = {
                "split": i,
                "test_start": split["test_start"],
                "test_end": split["test_end"],
                "n_test": len(y_test),
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "auc": roc_auc_score(y_test, probs),
                "profit_factor": profit_factor,
                "best_iteration": model.best_iteration,
            }
            all_metrics.append(metrics)

            logger.info(
                "Split %d: Precision=%.3f, Recall=%.3f, F1=%.3f, AUC=%.3f",
                i + 1, metrics["precision"], metrics["recall"],
                metrics["f1"], metrics["auc"]
            )

            # Accumulate feature importance
            feature_importance_sum += model.feature_importance(importance_type="gain")

            # Store predictions
            for j, idx in enumerate(test_idx):
                all_predictions.append({
                    "idx": int(idx),
                    "prob": float(probs[j]),
                    "actual": int(y_test[j]),
                    "split": i,
                })

        # Aggregate metrics
        aggregate = self._aggregate_metrics(all_metrics)

        # Train final model on all data
        logger.info("Training final model on all data...")
        pos_weight = (labels == 0).sum() / max((labels == 1).sum(), 1)
        params = self.params.copy()
        params["scale_pos_weight"] = pos_weight

        train_data = lgb.Dataset(tabular, label=labels, feature_name=feature_names)
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,  # Fixed rounds for final model
        )

        # Save model
        if save_path is None:
            save_path = "models/breakout_lgbm.txt"

        final_model.save_model(save_path)
        logger.info("Model saved to %s", save_path)

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance_sum / len(splits),
        }).sort_values("importance", ascending=False)

        logger.info("\nFeature Importance (top 10):")
        for _, row in importance_df.head(10).iterrows():
            logger.info("  %s: %.2f", row["feature"], row["importance"])

        # Log to database
        self._log_training_run(version, aggregate, save_path)

        logger.info("Training complete. Aggregate metrics: %s", aggregate)

        return {
            "version": version,
            "model_path": save_path,
            "metrics": aggregate,
            "split_metrics": all_metrics,
            "predictions": all_predictions,
            "feature_importance": importance_df.to_dict("records"),
        }

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate metrics across splits."""
        agg = {}
        for key in ["accuracy", "precision", "recall", "f1", "auc", "profit_factor"]:
            values = [m.get(key, 0) for m in metrics_list]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))

        agg["n_splits"] = len(metrics_list)
        agg["total_samples"] = sum(m["n_test"] for m in metrics_list)

        return agg

    def _log_training_run(self, version: str, metrics: dict, model_path: str):
        """Log training run to database."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_runs
                (model_type, model_version, precision_score, recall_score,
                 f1_score, accuracy, n_train_samples, hyperparameters, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "lightgbm",
                    version,
                    metrics.get("precision_mean", 0),
                    metrics.get("recall_mean", 0),
                    metrics.get("f1_mean", 0),
                    metrics.get("accuracy_mean", 0),
                    metrics.get("total_samples", 0),
                    str(self.params),
                    model_path,
                ),
            )
            conn.commit()


def train_lightgbm(
    version: Optional[str] = None,
    save_path: Optional[str] = None,
    label_strategy: Optional[str] = None,
) -> dict:
    """Train LightGBM model.

    Args:
        version: Model version string.
        save_path: Path to save model.
        label_strategy: Which outcome column to use ('asym_20_7', 'asym_15_10', 'sym_10').

    Returns:
        Training results dict.
    """
    trainer = LightGBMTrainer(label_strategy=label_strategy)
    return trainer.train(version=version, save_path=save_path)
