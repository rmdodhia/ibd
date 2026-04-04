"""Hybrid model combining CNN features with tabular features.

CNN backbone extracts shape features from raw price series.
Concatenated with tabular features for final classification.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scanner.config import get

logger = logging.getLogger(__name__)


class BreakoutDataset(Dataset):
    """PyTorch dataset for breakout prediction."""

    def __init__(
        self,
        price_series: np.ndarray,
        tabular_features: np.ndarray,
        labels: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            price_series: Shape (N, lookback_days, n_channels).
            tabular_features: Shape (N, n_features).
            labels: Shape (N,).
        """
        self.price_series = torch.FloatTensor(price_series)
        self.tabular = torch.FloatTensor(tabular_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.price_series[idx], self.tabular[idx], self.labels[idx]


class CNNBackbone(nn.Module):
    """1D CNN for extracting shape features from price series."""

    def __init__(
        self,
        in_channels: int = 5,
        embedding_dim: int = 128,
    ):
        """Initialize CNN backbone.

        Args:
            in_channels: Number of input channels (price series features).
            embedding_dim: Output embedding dimension.
        """
        super().__init__()

        # Conv layers with increasing receptive field
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, channels).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        # Transpose to (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, 2)

        # Global pooling
        x = self.pool(x).squeeze(-1)

        # FC layer
        x = self.dropout(x)
        x = F.relu(self.fc(x))

        return x


class TabularBranch(nn.Module):
    """MLP for processing tabular features."""

    def __init__(
        self,
        in_features: int = 21,
        embedding_dim: int = 32,
    ):
        """Initialize tabular branch.

        Args:
            in_features: Number of input features.
            embedding_dim: Output embedding dimension.
        """
        super().__init__()

        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class BreakoutPredictor(nn.Module):
    """Hybrid CNN + Tabular model for breakout prediction."""

    def __init__(
        self,
        price_channels: int = 5,
        tabular_features: int = 21,
        cnn_embedding: int = 128,
        tabular_embedding: int = 32,
    ):
        """Initialize model.

        Args:
            price_channels: Number of price series channels.
            tabular_features: Number of tabular features.
            cnn_embedding: CNN output embedding dimension.
            tabular_embedding: Tabular branch embedding dimension.
        """
        super().__init__()

        self.cnn = CNNBackbone(price_channels, cnn_embedding)
        self.tabular = TabularBranch(tabular_features, tabular_embedding)

        combined_dim = cnn_embedding + tabular_embedding

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self, price_series: torch.Tensor, tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            price_series: Shape (batch, seq_len, channels).
            tabular_features: Shape (batch, n_features).

        Returns:
            Logits tensor of shape (batch, 1).
        """
        cnn_out = self.cnn(price_series)
        tab_out = self.tabular(tabular_features)

        combined = torch.cat([cnn_out, tab_out], dim=1)
        out = self.head(combined)

        return out

    def predict_proba(
        self, price_series: torch.Tensor, tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict probabilities.

        Returns:
            Probability tensor of shape (batch,).
        """
        logits = self.forward(price_series, tabular_features)
        return torch.sigmoid(logits).squeeze(-1)


def create_model(
    price_channels: int = 5,
    tabular_features: int = 21,
    device: str = "cpu",
) -> BreakoutPredictor:
    """Create a new model instance.

    Args:
        price_channels: Number of price series channels.
        tabular_features: Number of tabular features.
        device: Device to put model on.

    Returns:
        BreakoutPredictor model.
    """
    model = BreakoutPredictor(
        price_channels=price_channels,
        tabular_features=tabular_features,
    )
    return model.to(device)


def save_model(model: BreakoutPredictor, path: str, metadata: Optional[dict] = None) -> None:
    """Save model to disk.

    Args:
        model: Model to save.
        path: Save path.
        metadata: Optional metadata to save alongside.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(state, path)
    logger.info("Model saved to %s", path)


def load_model(path: str, device: str = "cpu") -> Tuple[BreakoutPredictor, dict]:
    """Load model from disk.

    Args:
        path: Model path.
        device: Device to load model to.

    Returns:
        Tuple of (model, metadata).
    """
    state = torch.load(path, map_location=device, weights_only=True)

    model = create_model(device=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, state.get("metadata", {})


def get_default_model_path() -> Path:
    """Get default model save path."""
    return Path("models") / "breakout_predictor.pt"
