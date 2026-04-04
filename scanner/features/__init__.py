"""Feature extraction modules for pattern analysis."""

from scanner.features.pattern_features import extract_pattern_features
from scanner.features.volume_features import extract_volume_features
from scanner.features.rs_features import extract_rs_features
from scanner.features.fundamental_features import extract_fundamental_features
from scanner.features.market_features import extract_market_features

__all__ = [
    "extract_pattern_features",
    "extract_volume_features",
    "extract_rs_features",
    "extract_fundamental_features",
    "extract_market_features",
    "extract_all_features",
]


def extract_all_features(
    symbol: str,
    stock_df,
    index_df,
    base_start_date: str,
    base_end_date: str,
    breakout_date: str,
    pattern_metadata: dict = None,
) -> dict:
    """Extract all features for a breakout pattern.

    Args:
        symbol: Stock ticker symbol.
        stock_df: Stock price DataFrame.
        index_df: Index (S&P 500) price DataFrame.
        base_start_date: Pattern base start date.
        base_end_date: Pattern base end date.
        breakout_date: Breakout date.
        pattern_metadata: Optional metadata from pattern detector.

    Returns:
        Dict containing all feature categories.
    """
    if pattern_metadata is None:
        pattern_metadata = {}

    features = {}

    # Pattern geometry features
    pattern_feats = extract_pattern_features(
        stock_df, base_start_date, base_end_date, pattern_metadata
    )
    features.update(pattern_feats)

    # Volume features
    volume_feats = extract_volume_features(
        stock_df, base_start_date, base_end_date, breakout_date
    )
    features.update(volume_feats)

    # Relative strength features
    rs_feats = extract_rs_features(stock_df, index_df, breakout_date)
    features.update(rs_feats)

    # Fundamental features
    fundamental_feats = extract_fundamental_features(symbol, breakout_date)
    features.update(fundamental_feats)

    # Market context features
    market_feats = extract_market_features(stock_df, index_df, breakout_date)
    features.update(market_feats)

    return features
