"""Tests for feature extraction modules."""

import numpy as np
import pandas as pd
import pytest

from scanner.features import (
    extract_pattern_features,
    extract_volume_features,
    extract_rs_features,
    extract_market_features,
    extract_all_features,
)


def create_price_df(n_days: int = 200, start_price: float = 100) -> pd.DataFrame:
    """Create a sample price DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    prices = [start_price + i * 0.1 + np.random.uniform(-1, 1) for i in range(n_days)]

    return pd.DataFrame({
        "date": dates,
        "open": [p - 0.5 for p in prices],
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000000 + np.random.randint(-100000, 100000) for _ in range(n_days)],
    })


def create_index_df(n_days: int = 200) -> pd.DataFrame:
    """Create a sample index DataFrame for testing."""
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    prices = [3000 + i * 0.5 + np.random.uniform(-5, 5) for i in range(n_days)]

    return pd.DataFrame({
        "date": dates,
        "close": prices,
        "volume": [5000000000] * n_days,
    })


class TestPatternFeatures:
    """Tests for pattern feature extraction."""

    def test_extract_pattern_features(self):
        """Test pattern feature extraction."""
        df = create_price_df()
        features = extract_pattern_features(
            df,
            base_start_date="2020-02-01",
            base_end_date="2020-06-01",
            pattern_metadata={"handle_depth_pct": 5.0},
        )

        assert "base_depth_pct" in features
        assert "base_duration_weeks" in features
        assert "base_symmetry" in features
        assert "handle_depth_pct" in features
        assert "tightness_score" in features

        assert features["handle_depth_pct"] == 5.0
        assert 0 <= features["base_symmetry"] <= 1
        assert 0 <= features["tightness_score"] <= 1


class TestVolumeFeatures:
    """Tests for volume feature extraction."""

    def test_extract_volume_features(self):
        """Test volume feature extraction."""
        df = create_price_df()
        features = extract_volume_features(
            df,
            base_start_date="2020-02-01",
            base_end_date="2020-06-01",
            breakout_date="2020-06-01",
        )

        assert "breakout_volume_ratio" in features
        assert "volume_trend_in_base" in features
        assert "up_down_volume_ratio" in features
        assert "volume_dry_up" in features

        assert features["breakout_volume_ratio"] > 0
        assert features["up_down_volume_ratio"] > 0


class TestRSFeatures:
    """Tests for relative strength feature extraction."""

    def test_extract_rs_features(self):
        """Test RS feature extraction."""
        stock_df = create_price_df()
        index_df = create_index_df()

        features = extract_rs_features(
            stock_df,
            index_df,
            breakout_date="2020-08-01",
        )

        assert "rs_line_slope_4wk" in features
        assert "rs_line_slope_12wk" in features
        assert "rs_new_high" in features
        assert "rs_rank_percentile" in features

        assert isinstance(features["rs_new_high"], bool)
        assert 0 <= features["rs_rank_percentile"] <= 100


class TestMarketFeatures:
    """Tests for market context feature extraction."""

    def test_extract_market_features(self):
        """Test market feature extraction."""
        stock_df = create_price_df()
        index_df = create_index_df()

        features = extract_market_features(
            stock_df,
            index_df,
            breakout_date="2020-08-01",
        )

        assert "sp500_above_200dma" in features
        assert "sp500_trend_4wk" in features
        assert "price_vs_50dma" in features
        assert "price_vs_200dma" in features

        assert isinstance(features["sp500_above_200dma"], bool)


class TestAllFeatures:
    """Tests for combined feature extraction."""

    def test_extract_all_features(self):
        """Test extracting all features."""
        stock_df = create_price_df()
        index_df = create_index_df()

        features = extract_all_features(
            symbol="TEST",
            stock_df=stock_df,
            index_df=index_df,
            base_start_date="2020-02-01",
            base_end_date="2020-06-01",
            breakout_date="2020-06-01",
        )

        # Should have features from all categories
        assert "base_depth_pct" in features  # Pattern
        assert "breakout_volume_ratio" in features  # Volume
        assert "rs_line_slope_4wk" in features  # RS
        assert "sp500_above_200dma" in features  # Market
        # Note: fundamental features require database access
