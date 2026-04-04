"""Reusable ML training pipeline.

Handles data loading, walk-forward splits, model training,
evaluation, comparison, and saving. Same pipeline for initial
training and periodic retraining.

Optimizations:
- Automatic GPU detection (CUDA, MPS)
- Mixed precision training (AMP) for ~2x speedup on GPU
- Parallel data loading with multiple workers
- Pinned memory for faster CPU->GPU transfer
- Gradient clipping for stability

Usage:
    python -m scanner.train --model hybrid
    python -m scanner.train --model hybrid --compare-to v1.0
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from scanner.config import get
from scanner.db import get_cursor
from scanner.models.data_prep import (
    prepare_cnn_dataset,
    create_walk_forward_splits,
    get_feature_names,
)
from scanner.models.hybrid_model import (
    BreakoutPredictor,
    BreakoutDataset,
    create_model,
    save_model,
    load_model,
    get_default_model_path,
)

logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Get optimal number of DataLoader workers."""
    cpu_count = os.cpu_count() or 1
    # Use half of CPUs, max 8, min 0
    return min(max(cpu_count // 2, 0), 8)


class TrainingPipeline:
    """Training pipeline for breakout prediction models.

    Features:
    - Automatic GPU detection (CUDA, Apple MPS)
    - Mixed precision training for faster GPU training
    - Parallel data loading
    - Early stopping with patience
    - Learning rate scheduling
    """

    def __init__(
        self,
        model_type: str = "hybrid",
        device: Optional[str] = None,
        use_amp: bool = True,
        num_workers: Optional[int] = None,
    ):
        """Initialize training pipeline.

        Args:
            model_type: Model type ('hybrid', 'cnn', 'lightgbm').
            device: Device to train on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            use_amp: Use automatic mixed precision (faster on GPU).
            num_workers: DataLoader workers. Auto-detected if None.
        """
        self.model_type = model_type

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                # Log GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("GPU detected: %s (%.1f GB)", gpu_name, gpu_mem)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Apple MPS (Metal) detected")
            else:
                device = "cpu"
                logger.info("No GPU detected, using CPU")

        self.device = device
        logger.info("Using device: %s", self.device)

        # Mixed precision only works on CUDA
        self.use_amp = use_amp and self.device == "cuda"
        if self.use_amp:
            logger.info("Mixed precision training (AMP) enabled")

        # Parallel data loading
        if num_workers is None:
            num_workers = get_optimal_workers() if self.device != "mps" else 0
            # MPS has issues with multiprocessing
        self.num_workers = num_workers
        logger.info("DataLoader workers: %d", self.num_workers)

        # Pin memory for faster transfer (only for CUDA)
        self.pin_memory = self.device == "cuda"

        # Training config from YAML
        self.epochs = get("training.cnn.epochs", 100)
        self.batch_size = get("training.cnn.batch_size", 64)
        self.patience = get("training.cnn.early_stopping_patience", 10)
        self.learning_rate = get("training.cnn.learning_rate", 0.001)
        self.weight_decay = get("training.cnn.weight_decay", 1e-5)
        self.grad_clip = get("training.cnn.gradient_clip", 1.0)

    def train(
        self,
        version: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> dict:
        """Run full training pipeline with walk-forward validation.

        Args:
            version: Model version string.
            save_path: Path to save model.

        Returns:
            Dict with training results and metrics.
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        logger.info("Starting training pipeline for version %s", version)

        # Prepare dataset
        price_series, tabular, labels, metadata_df = prepare_cnn_dataset()
        if len(labels) == 0:
            logger.error("No training data available")
            return {"error": "No data"}

        logger.info(
            "Dataset: %d samples, %d success (%.1f%%)",
            len(labels),
            np.sum(labels == 1),
            np.mean(labels) * 100,
        )

        # Create walk-forward splits
        splits = create_walk_forward_splits(metadata_df)
        if not splits:
            logger.error("Could not create valid walk-forward splits")
            return {"error": "No valid splits"}

        # Train on each split and collect metrics
        all_metrics = []
        all_predictions = []

        for i, split in enumerate(splits):
            logger.info(
                "Split %d/%d: Train until %s, Test %s to %s",
                i + 1,
                len(splits),
                split["train_end"],
                split["test_start"],
                split["test_end"],
            )

            # Get split data
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            X_train_price = price_series[train_idx]
            X_train_tab = tabular[train_idx]
            y_train = labels[train_idx]

            X_test_price = price_series[test_idx]
            X_test_tab = tabular[test_idx]
            y_test = labels[test_idx]

            # Train model
            model, train_metrics = self._train_single_split(
                X_train_price, X_train_tab, y_train,
                X_test_price, X_test_tab, y_test,
            )

            # Evaluate on test set
            test_metrics = self._evaluate(model, X_test_price, X_test_tab, y_test)
            test_metrics["split"] = i
            test_metrics["test_start"] = split["test_start"]
            test_metrics["test_end"] = split["test_end"]

            all_metrics.append(test_metrics)

            # Store predictions for later analysis
            probs = self._predict_proba(model, X_test_price, X_test_tab)
            for j, idx in enumerate(test_idx):
                all_predictions.append({
                    "idx": int(idx),
                    "prob": float(probs[j]),
                    "actual": int(y_test[j]),
                    "split": i,
                })

            # Clear GPU memory between splits
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Aggregate metrics across splits
        aggregate = self._aggregate_metrics(all_metrics)

        # Train final model on all data
        logger.info("Training final model on all data...")
        final_model, _ = self._train_single_split(
            price_series, tabular, labels,
            price_series[:100], tabular[:100], labels[:100],  # Small validation set
        )

        # Save model
        if save_path is None:
            save_path = str(get_default_model_path())

        metadata = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "n_samples": int(len(labels)),
            "n_splits": len(splits),
            "metrics": aggregate,
            "feature_names": get_feature_names(),
            "device": self.device,
            "use_amp": self.use_amp,
        }

        save_model(final_model, save_path, metadata)

        # Log training run to database
        self._log_training_run(version, aggregate, save_path)

        logger.info("Training complete. Aggregate metrics: %s", aggregate)

        return {
            "version": version,
            "model_path": save_path,
            "metrics": aggregate,
            "split_metrics": all_metrics,
            "predictions": all_predictions,
        }

    def _train_single_split(
        self,
        X_train_price: np.ndarray,
        X_train_tab: np.ndarray,
        y_train: np.ndarray,
        X_val_price: np.ndarray,
        X_val_tab: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[BreakoutPredictor, dict]:
        """Train model on a single split.

        Returns:
            Tuple of (trained_model, training_metrics).
        """
        # Create data loaders with optimizations
        train_dataset = BreakoutDataset(X_train_price, X_train_tab, y_train)
        val_dataset = BreakoutDataset(X_val_price, X_val_tab, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

        # Create model
        n_channels = X_train_price.shape[2]
        n_features = X_train_tab.shape[1]
        model = create_model(
            price_channels=n_channels,
            tabular_features=n_features,
            device=self.device,
        )

        # Class weight for imbalanced data
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Mixed precision scaler
        scaler = GradScaler(enabled=self.use_amp)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0
            for price, tab, label in train_loader:
                price = price.to(self.device, non_blocking=True)
                tab = tab.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # Mixed precision forward pass
                with autocast(enabled=self.use_amp):
                    output = model(price, tab).squeeze(-1)
                    loss = criterion(output, label)

                # Scaled backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for price, tab, label in val_loader:
                    price = price.to(self.device, non_blocking=True)
                    tab = tab.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)

                    with autocast(enabled=self.use_amp):
                        output = model(price, tab).squeeze(-1)
                        loss = criterion(output, label)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        return model, {"best_val_loss": best_val_loss}

    def _evaluate(
        self,
        model: BreakoutPredictor,
        X_price: np.ndarray,
        X_tab: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Evaluate model on a dataset.

        Returns:
            Dict with evaluation metrics.
        """
        probs = self._predict_proba(model, X_price, X_tab)
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "n_samples": len(y),
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }

        if len(np.unique(y)) > 1:
            metrics["auc"] = float(roc_auc_score(y, probs))
        else:
            metrics["auc"] = 0.0

        # Profit factor (sum of wins / sum of losses)
        # Assuming equal position sizes
        wins = np.sum((preds == 1) & (y == 1))
        losses = np.sum((preds == 1) & (y == 0))
        metrics["profit_factor"] = float(wins / max(losses, 1))

        return metrics

    def _predict_proba(
        self,
        model: BreakoutPredictor,
        X_price: np.ndarray,
        X_tab: np.ndarray,
    ) -> np.ndarray:
        """Get probability predictions."""
        model.eval()
        dataset = BreakoutDataset(X_price, X_tab, np.zeros(len(X_price)))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        all_probs = []
        with torch.no_grad():
            for price, tab, _ in loader:
                price = price.to(self.device, non_blocking=True)
                tab = tab.to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    probs = model.predict_proba(price, tab)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict:
        """Aggregate metrics across splits."""
        if not metrics_list:
            return {}

        keys = ["accuracy", "precision", "recall", "f1", "auc", "profit_factor"]
        aggregate = {}

        for key in keys:
            values = [m.get(key, 0) for m in metrics_list]
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))

        aggregate["n_splits"] = len(metrics_list)
        aggregate["total_samples"] = sum(m.get("n_samples", 0) for m in metrics_list)

        return aggregate

    def _log_training_run(self, version: str, metrics: dict, model_path: str) -> None:
        """Log training run to database."""
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs
                (model_type, model_version, precision_score, recall_score,
                 f1_score, accuracy, n_train_samples, hyperparameters, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.model_type,
                    version,
                    metrics.get("precision_mean", 0),
                    metrics.get("recall_mean", 0),
                    metrics.get("f1_mean", 0),
                    metrics.get("accuracy_mean", 0),
                    metrics.get("total_samples", 0),
                    json.dumps({
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                        "device": self.device,
                        "use_amp": self.use_amp,
                    }),
                    model_path,
                ),
            )


def train_model(
    model_type: str = "hybrid",
    version: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict:
    """Convenience function to train a model.

    Args:
        model_type: Model type to train.
        version: Model version string.
        save_path: Path to save model.

    Returns:
        Training results dict.
    """
    pipeline = TrainingPipeline(model_type=model_type)
    return pipeline.train(version=version, save_path=save_path)
