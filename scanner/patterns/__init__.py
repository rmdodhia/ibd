"""Chart pattern detectors."""

from scanner.patterns.base_detector import BaseDetector, DetectedPattern
from scanner.patterns.cup_with_handle import CupWithHandleDetector
from scanner.patterns.double_bottom import DoubleBottomDetector
from scanner.patterns.flat_base import FlatBaseDetector

__all__ = [
    "BaseDetector",
    "DetectedPattern",
    "CupWithHandleDetector",
    "DoubleBottomDetector",
    "FlatBaseDetector",
]


def get_enabled_detectors() -> list[BaseDetector]:
    """Get list of all enabled pattern detectors.

    Returns:
        List of detector instances for enabled patterns.
    """
    from scanner.config import get

    detectors = []

    if get("patterns.cup_with_handle.enabled", True):
        detectors.append(CupWithHandleDetector())

    if get("patterns.double_bottom.enabled", True):
        detectors.append(DoubleBottomDetector())

    if get("patterns.flat_base.enabled", True):
        detectors.append(FlatBaseDetector())

    return detectors


def classify_pattern(df, symbol: str, breakout_date: str) -> tuple[str, float, dict]:
    """Classify the pattern preceding a breakout.

    Runs all enabled detectors and returns the best match.

    Args:
        df: Price DataFrame.
        symbol: Stock ticker.
        breakout_date: Date of the breakout.

    Returns:
        Tuple of (pattern_type, confidence, metadata).
        Returns ("unclassified", 0, {}) if no pattern matches.
    """
    import pandas as pd

    # Filter data up to breakout date
    df = df[df["date"] <= pd.to_datetime(breakout_date)]
    if len(df) < 50:
        return "unclassified", 0.0, {}

    best_type = "unclassified"
    best_confidence = 0.0
    best_metadata = {}

    for detector in get_enabled_detectors():
        try:
            patterns = detector.detect(symbol, df)
            for p in patterns:
                # Prefer patterns that end near the breakout date
                days_diff = abs(
                    (pd.to_datetime(p.base_end_date) - pd.to_datetime(breakout_date)).days
                )
                if days_diff <= 10 and p.confidence > best_confidence:
                    best_type = p.pattern_type
                    best_confidence = p.confidence
                    best_metadata = p.metadata
        except Exception:
            continue

    return best_type, best_confidence, best_metadata
