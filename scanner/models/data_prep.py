"""Data preparation for CNN model training.

Prepares price series tensors and tabular features for the hybrid model.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from scanner.config import get
from scanner.db import get_connection
from scanner.data_pipeline import get_price_data, get_index_data
from scanner.labeler import get_labeled_data

logger = logging.getLogger(__name__)


def prepare_cnn_dataset(
    lookback_days: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare full dataset for CNN training.

    Args:
        lookback_days: Days of price history per sample. Defaults to config.

    Returns:
        Tuple of (price_series, tabular_features, labels, metadata_df).
        - price_series: shape (N, lookback_days, n_channels)
        - tabular_features: shape (N, n_features)
        - labels: shape (N,) binary labels
        - metadata_df: DataFrame with sample metadata for analysis
    """
    if lookback_days is None:
        lookback_days = get("training.cnn.lookback_days", 200)

    # Get labeled data
    labeled_df = get_labeled_data()
    if labeled_df.empty:
        logger.error("No labeled data available. Run labeler first.")
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()

    logger.info("Processing %d labeled patterns...", len(labeled_df))

    # Get index data for RS calculations
    rs_benchmark = get("features.rs_benchmark", "^GSPC")
    index_df = get_index_data(rs_benchmark)

    # Process each sample
    price_series_list = []
    tabular_list = []
    labels_list = []
    metadata_list = []

    for idx, row in labeled_df.iterrows():
        result = _prepare_single_sample(
            row, lookback_days, index_df
        )
        if result is not None:
            price_series, tabular, label, meta = result
            price_series_list.append(price_series)
            tabular_list.append(tabular)
            labels_list.append(label)
            metadata_list.append(meta)

    if not price_series_list:
        logger.error("No valid samples could be prepared.")
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()

    # Stack into arrays
    price_series = np.stack(price_series_list, axis=0)
    tabular = np.stack(tabular_list, axis=0)
    labels = np.array(labels_list)
    metadata_df = pd.DataFrame(metadata_list)

    logger.info(
        "Prepared %d samples: %d success, %d failure",
        len(labels),
        np.sum(labels == 1),
        np.sum(labels == 0),
    )

    return price_series, tabular, labels, metadata_df


def _prepare_single_sample(
    row: pd.Series,
    lookback_days: int,
    index_df: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, dict]]:
    """Prepare a single sample.

    Returns:
        Tuple of (price_series, tabular_features, label, metadata) or None.
    """
    symbol = row["symbol"]
    breakout_date = row["pivot_date"]
    outcome = row["outcome"]

    # Get price data
    stock_df = get_price_data(symbol)
    if stock_df.empty:
        return None

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    breakout_dt = pd.to_datetime(breakout_date)

    # Find breakout index
    mask = stock_df["date"] <= breakout_dt
    stock_df_filtered = stock_df[mask]

    if len(stock_df_filtered) < lookback_days:
        return None

    # Extract lookback window
    window_df = stock_df_filtered.tail(lookback_days).reset_index(drop=True)

    # Prepare price series tensor
    price_series = _extract_price_series(window_df, index_df)
    if price_series is None:
        return None

    # Extract tabular features
    tabular = _extract_tabular_features(row)

    # Label: 1 = success, 0 = failure
    label = 1 if outcome == "success" else 0

    # Metadata for analysis
    metadata = {
        "symbol": symbol,
        "breakout_date": breakout_date,
        "outcome": outcome,
        "pattern_type": row.get("pattern_type", "unknown"),
    }

    return price_series, tabular, label, metadata


def _extract_price_series(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Extract normalized price series with multiple channels.

    Channels:
    1. close_norm: Close price normalized to start at 1
    2. volume_norm: Volume normalized by 50-day average
    3. rs_line: Relative strength line vs index
    4. ma50_ratio: Price / 50-day MA
    5. ma200_ratio: Price / 200-day MA

    Returns:
        Array of shape (lookback_days, 5) or None.
    """
    if len(df) < 50:
        return None

    close = df["close"].values
    volume = df["volume"].values

    # 1. Normalized close (start at 1)
    close_norm = close / close[0] if close[0] > 0 else close

    # 2. Normalized volume (by 50-day rolling avg)
    vol_avg = pd.Series(volume).rolling(50, min_periods=10).mean().fillna(volume[0]).values
    volume_norm = volume / vol_avg
    volume_norm = np.clip(volume_norm, 0, 5)  # Cap at 5x average

    # 3. RS line (if index data available)
    if not index_df.empty and len(df) > 0:
        # Merge on date
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        index_copy = index_df.copy()
        index_copy["date"] = pd.to_datetime(index_copy["date"])

        merged = pd.merge(
            df_copy[["date", "close"]],
            index_copy[["date", "close"]],
            on="date",
            suffixes=("", "_idx"),
            how="left",
        )

        if "close_idx" in merged.columns and merged["close_idx"].notna().sum() > 0:
            merged["close_idx"] = merged["close_idx"].ffill().bfill()
            rs = merged["close"].values / merged["close_idx"].values
            rs_line = rs / rs[0] if rs[0] > 0 else rs
        else:
            rs_line = np.ones(len(close))
    else:
        rs_line = np.ones(len(close))

    # 4. MA50 ratio
    ma50 = pd.Series(close).rolling(50, min_periods=10).mean().fillna(close[0]).values
    ma50_ratio = close / ma50
    ma50_ratio = np.clip(ma50_ratio, 0.5, 1.5)

    # 5. MA200 ratio (use MA50 if not enough data)
    if len(close) >= 200:
        ma200 = pd.Series(close).rolling(200, min_periods=50).mean().fillna(close[0]).values
    else:
        ma200 = ma50  # Fallback
    ma200_ratio = close / ma200
    ma200_ratio = np.clip(ma200_ratio, 0.5, 1.5)

    # Stack channels
    series = np.stack([close_norm, volume_norm, rs_line, ma50_ratio, ma200_ratio], axis=1)

    return series.astype(np.float32)


def _extract_tabular_features(row: pd.Series) -> np.ndarray:
    """Extract tabular features from labeled data row.

    Returns:
        Array of tabular features.
    """
    feature_names = [
        "base_depth_pct",
        "base_duration_weeks",
        "base_symmetry",
        "handle_depth_pct",
        "tightness_score",
        "breakout_volume_ratio",
        "volume_trend_in_base",
        "up_down_volume_ratio",
        "rs_line_slope_4wk",
        "rs_line_slope_12wk",
        "rs_rank_percentile",
        "eps_latest_yoy_growth",
        "eps_acceleration",
        "revenue_latest_yoy_growth",
        "institutional_pct",
        "market_cap_log",
        "sp500_trend_4wk",
        "price_vs_50dma",
        "price_vs_200dma",
    ]

    features = []
    for name in feature_names:
        val = row.get(name, 0)
        if pd.isna(val):
            val = 0
        features.append(float(val))

    # Add boolean features
    rs_new_high = 1.0 if row.get("rs_new_high", False) else 0.0
    sp500_above_200dma = 1.0 if row.get("sp500_above_200dma", True) else 0.0
    features.extend([rs_new_high, sp500_above_200dma])

    return np.array(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    """Get list of tabular feature names in order."""
    return [
        "base_depth_pct",
        "base_duration_weeks",
        "base_symmetry",
        "handle_depth_pct",
        "tightness_score",
        "breakout_volume_ratio",
        "volume_trend_in_base",
        "up_down_volume_ratio",
        "rs_line_slope_4wk",
        "rs_line_slope_12wk",
        "rs_rank_percentile",
        "eps_latest_yoy_growth",
        "eps_acceleration",
        "revenue_latest_yoy_growth",
        "institutional_pct",
        "market_cap_log",
        "sp500_trend_4wk",
        "price_vs_50dma",
        "price_vs_200dma",
        "rs_new_high",
        "sp500_above_200dma",
    ]


def create_walk_forward_splits(
    metadata_df: pd.DataFrame,
    embargo_weeks: int = 8,
) -> list[dict]:
    """Create walk-forward validation splits with embargo.

    Args:
        metadata_df: DataFrame with 'breakout_date' column.
        embargo_weeks: Weeks of embargo between train and test.

    Returns:
        List of split dicts with train_idx and test_idx arrays.
    """
    test_years = get("training.walk_forward_test_years", 1)
    embargo_days = embargo_weeks * 7

    metadata_df = metadata_df.copy()
    metadata_df["breakout_date"] = pd.to_datetime(metadata_df["breakout_date"])

    min_date = metadata_df["breakout_date"].min()
    max_date = metadata_df["breakout_date"].max()

    splits = []
    current_test_start = min_date + pd.Timedelta(days=3 * 365)  # Start testing after 3 years

    while current_test_start < max_date:
        test_end = current_test_start + pd.Timedelta(days=test_years * 365)

        # Train: all data before embargo
        train_end = current_test_start - pd.Timedelta(days=embargo_days)

        train_mask = metadata_df["breakout_date"] < train_end
        test_mask = (metadata_df["breakout_date"] >= current_test_start) & (
            metadata_df["breakout_date"] < test_end
        )

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= 100 and len(test_idx) >= 20:
            splits.append({
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": current_test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })

        current_test_start = test_end

    logger.info("Created %d walk-forward splits", len(splits))
    return splits
