"""Live scanner — detect current breakout candidates.

Usage:
    python -m scanner.scan
    python -m scanner.scan --symbol AAPL
    python -m scanner.scan --min-confidence 0.7
    python -m scanner.scan --output json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from scanner.config import get, get_price_range
from scanner.db import init_db, get_cursor
from scanner.universe import get_universe
from scanner.data_pipeline import get_price_data, get_index_data
from scanner.breakout_detector import detect_breakouts
from scanner.patterns import classify_pattern
from scanner.features import extract_all_features
from scanner.models.hybrid_model import load_model, get_default_model_path
from scanner.models.data_prep import _extract_price_series, _extract_tabular_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_candidates(
    symbol: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_results: Optional[int] = None,
) -> list[dict]:
    """Find current breakout candidates.

    Args:
        symbol: Optional single symbol to scan.
        min_confidence: Minimum confidence score (default from config).
        max_results: Maximum results to return (default from config).

    Returns:
        List of candidate dicts with symbol, confidence, pattern_type, etc.
    """
    if min_confidence is None:
        min_confidence = get("scanner.min_confidence", 0.6)
    if max_results is None:
        max_results = get("scanner.max_results", 50)

    init_db()

    # Load model
    model_path = get_default_model_path()
    if not model_path.exists():
        logger.error("No trained model found at %s. Run training first.", model_path)
        return []

    model, metadata = load_model(str(model_path))
    logger.info("Loaded model version: %s", metadata.get("version", "unknown"))

    # Get symbols to scan
    if symbol:
        symbols = [symbol]
    else:
        symbols = get_universe()
        if not symbols:
            logger.error("No symbols in universe.")
            return []

    # Get index data for RS calculations
    rs_benchmark = get("features.rs_benchmark", "^GSPC")
    index_df = get_index_data(rs_benchmark)

    candidates = []
    lookback_days = get("training.cnn.lookback_days", 200)

    logger.info("Scanning %d symbols...", len(symbols))

    for sym in tqdm(symbols, desc="Scanning", disable=symbol is not None):
        try:
            result = _scan_symbol(sym, model, index_df, lookback_days)
            if result and result["confidence"] >= min_confidence:
                candidates.append(result)
        except Exception as e:
            logger.debug("Error scanning %s: %s", sym, e)

    # Sort by confidence
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    # Limit results
    if len(candidates) > max_results:
        candidates = candidates[:max_results]

    return candidates


def _scan_symbol(
    symbol: str,
    model,
    index_df: pd.DataFrame,
    lookback_days: int,
) -> Optional[dict]:
    """Scan a single symbol for breakout setup.

    Returns:
        Dict with candidate info or None.
    """
    import torch

    # Get recent price data
    stock_df = get_price_data(symbol)
    if stock_df.empty or len(stock_df) < lookback_days + 50:
        return None

    stock_df["date"] = pd.to_datetime(stock_df["date"])

    # Check if approaching consolidation high
    recent = stock_df.tail(lookback_days + 50)
    if len(recent) < lookback_days:
        return None

    current_price = recent["close"].iloc[-1]
    recent_high_series, _ = get_price_range(recent)
    recent_high = recent_high_series.max()
    recent_50d = recent.tail(50)
    high_50d_series, _ = get_price_range(recent_50d)
    high_50d = high_50d_series.max()

    # Must be within 5% of 50-day high
    pct_from_high = ((recent_high - current_price) / recent_high) * 100
    if pct_from_high > 10:
        return None

    # Check volume
    vol_avg = recent.tail(50)["volume"].mean()
    vol_recent = recent.tail(5)["volume"].mean()
    vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 0

    # Prepare input for model
    window_df = recent.tail(lookback_days).reset_index(drop=True)
    price_series = _extract_price_series(window_df, index_df)
    if price_series is None:
        return None

    # Build tabular features (use most recent values)
    today = recent["date"].max().strftime("%Y-%m-%d")
    base_start = recent.iloc[-lookback_days]["date"].strftime("%Y-%m-%d")

    features = extract_all_features(
        symbol=symbol,
        stock_df=recent,
        index_df=index_df,
        base_start_date=base_start,
        base_end_date=today,
        breakout_date=today,
    )

    # Convert to tensor
    tabular = _build_tabular_from_features(features)

    # Get prediction
    price_tensor = torch.FloatTensor(price_series).unsqueeze(0)
    tabular_tensor = torch.FloatTensor(tabular).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        confidence = model.predict_proba(price_tensor, tabular_tensor).item()

    # Classify pattern
    pattern_type, pattern_conf, pattern_meta = classify_pattern(
        recent, symbol, today
    )

    return {
        "symbol": symbol,
        "confidence": float(confidence),
        "pattern_type": pattern_type,
        "pattern_confidence": float(pattern_conf),
        "current_price": float(current_price),
        "recent_high": float(recent_high),
        "pct_from_high": float(pct_from_high),
        "volume_ratio": float(vol_ratio),
        "scan_date": today,
    }


def _build_tabular_from_features(features: dict) -> np.ndarray:
    """Build tabular feature array from features dict."""
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

    values = []
    for name in feature_names:
        val = features.get(name, 0)
        if pd.isna(val):
            val = 0
        values.append(float(val))

    # Boolean features
    rs_new_high = 1.0 if features.get("rs_new_high", False) else 0.0
    sp500_above_200dma = 1.0 if features.get("sp500_above_200dma", True) else 0.0
    values.extend([rs_new_high, sp500_above_200dma])

    return np.array(values, dtype=np.float32)


def save_predictions(candidates: list[dict]) -> int:
    """Save scan results to predictions table.

    Returns:
        Number of predictions saved.
    """
    count = 0
    with get_cursor() as cur:
        for c in candidates:
            cur.execute(
                """
                INSERT INTO predictions
                (symbol, prediction_date, confidence_score, predicted_outcome)
                VALUES (?, ?, ?, ?)
                """,
                (
                    c["symbol"],
                    c["scan_date"],
                    c["confidence"],
                    "success" if c["confidence"] >= 0.6 else "uncertain",
                ),
            )
            count += 1

    return count


def format_output(candidates: list[dict], output_format: str = "table") -> str:
    """Format candidates for display.

    Args:
        candidates: List of candidate dicts.
        output_format: 'table', 'json', or 'csv'.

    Returns:
        Formatted string.
    """
    if not candidates:
        return "No candidates found."

    if output_format == "json":
        return json.dumps(candidates, indent=2)

    if output_format == "csv":
        df = pd.DataFrame(candidates)
        return df.to_csv(index=False)

    # Default: table format
    lines = [
        "",
        "=" * 80,
        "BREAKOUT CANDIDATES",
        "=" * 80,
        "",
        f"{'Symbol':<8} {'Conf':>6} {'Pattern':<18} {'Price':>10} {'From High':>10} {'Vol':>6}",
        "-" * 80,
    ]

    for c in candidates:
        lines.append(
            f"{c['symbol']:<8} {c['confidence']:>5.1%} {c['pattern_type']:<18} "
            f"${c['current_price']:>8.2f} {c['pct_from_high']:>9.1f}% {c['volume_ratio']:>5.1f}x"
        )

    lines.append("-" * 80)
    lines.append(f"Total: {len(candidates)} candidates")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scan for breakout candidates")
    parser.add_argument(
        "--symbol",
        type=str,
        help="Scan a single symbol",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence score (default: 0.6)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum results to show (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="table",
        choices=["table", "json", "csv"],
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save predictions to database",
    )
    args = parser.parse_args()

    candidates = find_candidates(
        symbol=args.symbol,
        min_confidence=args.min_confidence,
        max_results=args.max_results,
    )

    print(format_output(candidates, args.output))

    if args.save and candidates:
        count = save_predictions(candidates)
        print(f"\nSaved {count} predictions to database.")


if __name__ == "__main__":
    main()
