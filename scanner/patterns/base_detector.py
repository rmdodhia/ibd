"""Base pattern detector with shared peak/trough detection and smoothing.

All pattern detectors inherit from BaseDetector and implement detect().
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from scanner.config import get

logger = logging.getLogger(__name__)


@dataclass
class DetectedPattern:
    """A single detected chart pattern."""
    symbol: str
    pattern_type: str
    base_start_date: str
    base_end_date: str
    pivot_date: str
    pivot_price: float
    confidence: float
    metadata: dict  # pattern-specific details


class BaseDetector(ABC):
    """Abstract base class for all pattern detectors."""

    def __init__(self):
        self.sensitivity = get("patterns.sensitivity", 0.5)

    @abstractmethod
    def detect(self, symbol: str, df: pd.DataFrame) -> list[DetectedPattern]:
        """Scan a stock's price history for this pattern.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with columns [date, open, high, low, close, adj_close, volume],
                sorted by date ascending.

        Returns:
            List of detected patterns.
        """
        pass

    @staticmethod
    def find_peaks(prices: np.ndarray, order: int = 10) -> np.ndarray:
        """Find local maxima in price series.

        Args:
            prices: Array of prices.
            order: How many points on each side to compare. Higher = smoother.

        Returns:
            Array of indices where local maxima occur.
        """
        indices = argrelextrema(prices, np.greater_equal, order=order)[0]
        return indices

    @staticmethod
    def find_troughs(prices: np.ndarray, order: int = 10) -> np.ndarray:
        """Find local minima in price series.

        Args:
            prices: Array of prices.
            order: How many points on each side to compare. Higher = smoother.

        Returns:
            Array of indices where local minima occur.
        """
        indices = argrelextrema(prices, np.less_equal, order=order)[0]
        return indices

    @staticmethod
    def smooth(prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth price series with a simple moving average.

        Args:
            prices: Raw price array.
            window: SMA window size.

        Returns:
            Smoothed price array (same length, front-filled).
        """
        return pd.Series(prices).rolling(window=window, min_periods=1).mean().values

    @staticmethod
    def compute_depth_pct(peak_price: float, trough_price: float) -> float:
        """Calculate percentage decline from peak to trough.

        Args:
            peak_price: Price at the peak.
            trough_price: Price at the trough.

        Returns:
            Depth as a positive percentage (e.g., 25.0 for a 25% decline).
        """
        if peak_price == 0:
            return 0.0
        return ((peak_price - trough_price) / peak_price) * 100

    @staticmethod
    def trading_days_to_weeks(n_days: int) -> float:
        """Convert trading days to approximate weeks.

        Args:
            n_days: Number of trading days.

        Returns:
            Approximate number of weeks.
        """
        return n_days / 5.0

    def adjust_threshold(self, base_value: float, direction: str = "loose") -> float:
        """Adjust a threshold based on sensitivity setting.

        Args:
            base_value: The default threshold value.
            direction: 'loose' means higher sensitivity widens the threshold,
                       'tight' means higher sensitivity tightens it.

        Returns:
            Adjusted threshold value.
        """
        adjustment = self.sensitivity * 0.3  # ±30% at max sensitivity
        if direction == "loose":
            return base_value * (1 + adjustment)
        else:
            return base_value * (1 - adjustment)

    def check_prior_uptrend(
        self,
        df: pd.DataFrame,
        base_start_idx: int,
        min_advance_pct: float = 25.0,
        lookback_weeks: int = 26,
    ) -> tuple[bool, float]:
        """Check if there was a sufficient prior uptrend before the base.

        IBD bases should form after meaningful advances (typically 25-30%+).

        Args:
            df: Price DataFrame with 'close' column.
            base_start_idx: Index where the base starts (left lip).
            min_advance_pct: Minimum required advance percentage.
            lookback_weeks: How many weeks to look back for the prior low.

        Returns:
            Tuple of (has_uptrend: bool, advance_pct: float).
        """
        lookback_days = lookback_weeks * 5
        lookback_start = max(0, base_start_idx - lookback_days)

        if lookback_start >= base_start_idx:
            return False, 0.0

        prior_region = df.iloc[lookback_start:base_start_idx]
        if len(prior_region) < 10:
            return False, 0.0

        prior_low = prior_region["low"].min()
        base_start_price = df.iloc[base_start_idx]["close"]

        if prior_low <= 0:
            return False, 0.0

        advance_pct = ((base_start_price - prior_low) / prior_low) * 100

        return advance_pct >= min_advance_pct, advance_pct
