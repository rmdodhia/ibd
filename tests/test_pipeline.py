"""Tests for the ML training pipeline."""

import numpy as np
import pytest
import torch

from scanner.models.hybrid_model import (
    BreakoutPredictor,
    BreakoutDataset,
    CNNBackbone,
    TabularBranch,
    create_model,
)


class TestBreakoutPredictor:
    """Tests for the hybrid model architecture."""

    def test_cnn_backbone_forward(self):
        """Test CNN backbone forward pass."""
        backbone = CNNBackbone(in_channels=5, embedding_dim=128)

        # Input: (batch, seq_len, channels)
        x = torch.randn(4, 200, 5)
        out = backbone(x)

        assert out.shape == (4, 128)

    def test_tabular_branch_forward(self):
        """Test tabular branch forward pass."""
        branch = TabularBranch(in_features=21, embedding_dim=32)

        x = torch.randn(4, 21)
        out = branch(x)

        assert out.shape == (4, 32)

    def test_full_model_forward(self):
        """Test full hybrid model forward pass."""
        model = BreakoutPredictor(
            price_channels=5,
            tabular_features=21,
            cnn_embedding=128,
            tabular_embedding=32,
        )

        price_series = torch.randn(4, 200, 5)
        tabular = torch.randn(4, 21)

        logits = model(price_series, tabular)

        assert logits.shape == (4, 1)

    def test_predict_proba(self):
        """Test probability prediction."""
        model = BreakoutPredictor()
        model.eval()

        price_series = torch.randn(4, 200, 5)
        tabular = torch.randn(4, 21)

        probs = model.predict_proba(price_series, tabular)

        assert probs.shape == (4,)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)


class TestBreakoutDataset:
    """Tests for the PyTorch dataset."""

    def test_dataset_creation(self):
        """Test dataset initialization."""
        price_series = np.random.randn(100, 200, 5).astype(np.float32)
        tabular = np.random.randn(100, 21).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)

        dataset = BreakoutDataset(price_series, tabular, labels)

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        price_series = np.random.randn(10, 200, 5).astype(np.float32)
        tabular = np.random.randn(10, 21).astype(np.float32)
        labels = np.random.randint(0, 2, 10).astype(np.float32)

        dataset = BreakoutDataset(price_series, tabular, labels)

        p, t, l = dataset[0]

        assert p.shape == (200, 5)
        assert t.shape == (21,)
        assert l.shape == ()


class TestCreateModel:
    """Tests for model creation utility."""

    def test_create_model_default(self):
        """Test creating model with defaults."""
        model = create_model()

        assert isinstance(model, BreakoutPredictor)

    def test_create_model_custom(self):
        """Test creating model with custom parameters."""
        model = create_model(
            price_channels=3,
            tabular_features=10,
            device="cpu",
        )

        # Test forward pass with custom dimensions
        price_series = torch.randn(2, 200, 3)
        tabular = torch.randn(2, 10)

        logits = model(price_series, tabular)
        assert logits.shape == (2, 1)
