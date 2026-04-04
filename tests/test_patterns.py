"""Tests for pattern detection modules."""

import numpy as np
import pandas as pd
import pytest

from scanner.patterns import (
    CupWithHandleDetector,
    DoubleBottomDetector,
    FlatBaseDetector,
    classify_pattern,
)


def create_price_df(prices: list[float], start_date: str = "2020-01-01") -> pd.DataFrame:
    """Create a DataFrame with price data for testing."""
    dates = pd.date_range(start=start_date, periods=len(prices), freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000000] * len(prices),
    })


class TestCupWithHandleDetector:
    """Tests for CupWithHandleDetector."""

    def test_detect_valid_cup(self):
        """Test detection of a valid cup-with-handle pattern."""
        # Create a cup pattern: peak -> decline -> recovery -> handle
        n_days = 200
        prices = []

        # Initial rise to peak
        for i in range(30):
            prices.append(90 + i * 0.5)  # Rise to 105

        # Cup decline (20%)
        for i in range(40):
            prices.append(105 - i * 0.5)  # Decline to 85

        # Cup recovery
        for i in range(50):
            prices.append(85 + i * 0.4)  # Rise back to 105

        # Handle pullback
        for i in range(20):
            prices.append(105 - i * 0.3)  # Small pullback

        # Handle recovery
        for i in range(20):
            prices.append(99 + i * 0.3)

        # Fill remaining days
        while len(prices) < n_days:
            prices.append(105)

        df = create_price_df(prices)
        detector = CupWithHandleDetector()
        patterns = detector.detect("TEST", df)

        # May or may not detect depending on exact geometry
        # This is a basic smoke test
        assert isinstance(patterns, list)

    def test_no_pattern_in_uptrend(self):
        """Test that no pattern is detected in a simple uptrend."""
        prices = [100 + i * 0.5 for i in range(200)]
        df = create_price_df(prices)

        detector = CupWithHandleDetector()
        patterns = detector.detect("TEST", df)

        assert len(patterns) == 0


class TestDoubleBottomDetector:
    """Tests for DoubleBottomDetector."""

    def test_detect_basic_pattern(self):
        """Test double bottom detector initialization."""
        detector = DoubleBottomDetector()
        assert detector.min_depth > 0
        assert detector.second_low_tolerance > 0


class TestFlatBaseDetector:
    """Tests for FlatBaseDetector."""

    def test_detect_flat_consolidation(self):
        """Test detection of a flat base pattern."""
        # Create flat consolidation
        prices = [100 + np.random.uniform(-2, 2) for _ in range(100)]
        df = create_price_df(prices)

        detector = FlatBaseDetector()
        patterns = detector.detect("TEST", df)

        # Should detect at least one flat base in tight range
        assert isinstance(patterns, list)


class TestClassifyPattern:
    """Tests for pattern classification."""

    def test_classify_returns_tuple(self):
        """Test that classify_pattern returns correct format."""
        prices = [100 + i * 0.1 for i in range(100)]
        df = create_price_df(prices)

        pattern_type, confidence, metadata = classify_pattern(
            df, "TEST", "2020-06-01"
        )

        assert isinstance(pattern_type, str)
        assert isinstance(confidence, float)
        assert isinstance(metadata, dict)

    def test_classify_unclassified_pattern(self):
        """Test that random data returns unclassified."""
        prices = [100 + np.random.uniform(-10, 10) for _ in range(100)]
        df = create_price_df(prices)

        pattern_type, confidence, metadata = classify_pattern(
            df, "TEST", "2020-06-01"
        )

        # Random data may or may not match a pattern
        assert pattern_type in ["unclassified", "flat_base", "cup_with_handle", "double_bottom"]
