#!/usr/bin/env python3
"""Validate pattern detectors against known examples and measure success rates.

Validation approaches:
1. Known patterns: Test against documented breakouts from BreakoutDB, IBD, etc.
2. Success rate: Measure what % of detected patterns lead to successful breakouts
3. Quality filtering: Test if quality scoring improves success rate

Sources:
- BreakoutDB: https://breakoutdb.com/pattern/cup-with-handle/
- BreakoutDB: https://breakoutdb.com/pattern/double-bottom/
- IBD historical articles
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from scanner.patterns import CupWithHandleDetector, DoubleBottomDetector
from scanner.quality_score import compute_quality_score, get_quality_summary
from scanner.features import extract_all_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Known documented cup-with-handle patterns
KNOWN_CUP_WITH_HANDLE = [
    ("ZS", "2020-12-03", "BreakoutDB"),
    ("WDAY", "2020-08-26", "BreakoutDB"),
    ("AAPL", "2020-06-03", "MoneyShow - breakout near $327"),
    ("NFLX", "2020-04-15", "Admiral Markets - April 2020 breakout"),
    ("MSFT", "2020-06-05", "Post-COVID recovery cup"),
    ("NVDA", "2020-09-01", "Pre-split run-up"),
    ("CRM", "2020-08-26", "Salesforce breakout"),
    ("PYPL", "2020-07-06", "PayPal summer breakout"),
]

# Known documented double-bottom patterns
KNOWN_DOUBLE_BOTTOM = [
    ("PAYC", "2019-01-23", "BreakoutDB - +14% gain"),
    ("AAPL", "2019-01-30", "Post-2018 correction recovery"),
    ("MSFT", "2018-12-26", "Christmas 2018 bottom"),
    ("AMZN", "2019-01-04", "Q4 2018 double bottom"),
    ("GOOGL", "2018-12-24", "2018 correction bottom"),
    ("JPM", "2019-01-04", "Banking sector recovery"),
    ("BA", "2020-05-26", "COVID recovery W-pattern"),
]


def fetch_data_for_validation(symbol: str, breakout_date: str, lookback_days: int = 400) -> pd.DataFrame:
    """Fetch historical data around a known breakout date."""
    breakout = datetime.strptime(breakout_date, "%Y-%m-%d")
    start = breakout - timedelta(days=lookback_days)
    end = breakout + timedelta(days=60)

    try:
        data = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return pd.DataFrame()

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def fetch_index_data(period: str = "3y") -> pd.DataFrame:
    """Fetch S&P 500 index data."""
    try:
        data = yf.download("^GSPC", period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
        return data
    except Exception:
        return pd.DataFrame()


def validate_cup_with_handle():
    """Validate cup-with-handle detector against known examples."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING CUP-WITH-HANDLE DETECTOR")
    logger.info("=" * 60)

    detector = CupWithHandleDetector()
    found = 0
    total = len(KNOWN_CUP_WITH_HANDLE)

    for symbol, breakout_date, source in KNOWN_CUP_WITH_HANDLE:
        logger.info(f"\n{symbol} - Expected breakout: {breakout_date}")
        logger.info(f"  Source: {source}")

        df = fetch_data_for_validation(symbol, breakout_date)
        if df.empty:
            logger.info(f"  No data available")
            continue

        patterns = detector.detect(symbol, df)
        breakout_dt = datetime.strptime(breakout_date, "%Y-%m-%d")
        matched = False

        for p in patterns:
            pattern_end = datetime.strptime(p.base_end_date, "%Y-%m-%d")
            days_diff = abs((pattern_end - breakout_dt).days)

            if days_diff <= 30:
                logger.info(f"  FOUND: {p.base_start_date} to {p.base_end_date} (conf={p.confidence:.2f})")
                logger.info(f"     Depth: {p.metadata.get('depth_pct', 0):.1f}%, "
                           f"Recovery: {p.metadata.get('recovery_pct', 0):.1f}%")
                matched = True
                found += 1
                break

        if not matched:
            logger.info(f"  NOT FOUND (detected {len(patterns)} patterns, none near breakout date)")
            if patterns:
                closest = min(patterns, key=lambda p: abs((datetime.strptime(p.base_end_date, "%Y-%m-%d") - breakout_dt).days))
                logger.info(f"     Closest: {closest.base_end_date} ({abs((datetime.strptime(closest.base_end_date, '%Y-%m-%d') - breakout_dt).days)} days off)")

    logger.info(f"\nCup-with-handle: {found}/{total} known patterns detected ({found/total*100:.0f}%)")
    return found, total


def validate_double_bottom():
    """Validate double-bottom detector against known examples."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING DOUBLE-BOTTOM DETECTOR")
    logger.info("=" * 60)

    detector = DoubleBottomDetector()
    found = 0
    total = len(KNOWN_DOUBLE_BOTTOM)

    for symbol, breakout_date, source in KNOWN_DOUBLE_BOTTOM:
        logger.info(f"\n{symbol} - Expected breakout: {breakout_date}")
        logger.info(f"  Source: {source}")

        df = fetch_data_for_validation(symbol, breakout_date)
        if df.empty:
            logger.info(f"  No data available")
            continue

        patterns = detector.detect(symbol, df)
        breakout_dt = datetime.strptime(breakout_date, "%Y-%m-%d")
        matched = False

        for p in patterns:
            pattern_end = datetime.strptime(p.base_end_date, "%Y-%m-%d")
            days_diff = abs((pattern_end - breakout_dt).days)

            if days_diff <= 30:
                logger.info(f"  FOUND: {p.base_start_date} to {p.base_end_date} (conf={p.confidence:.2f})")
                logger.info(f"     Depth: {p.metadata.get('depth_pct', 0):.1f}%, "
                           f"Low diff: {p.metadata.get('low_diff_pct', 0):.1f}%")
                matched = True
                found += 1
                break

        if not matched:
            logger.info(f"  NOT FOUND (detected {len(patterns)} patterns, none near breakout date)")
            if patterns:
                closest = min(patterns, key=lambda p: abs((datetime.strptime(p.base_end_date, "%Y-%m-%d") - breakout_dt).days))
                logger.info(f"     Closest: {closest.base_end_date} ({abs((datetime.strptime(closest.base_end_date, '%Y-%m-%d') - breakout_dt).days)} days off)")

    logger.info(f"\nDouble-bottom: {found}/{total} known patterns detected ({found/total*100:.0f}%)")
    return found, total


def measure_success_rates():
    """Measure success rates with and without quality filtering."""
    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS RATE ANALYSIS")
    logger.info("=" * 60)

    test_symbols = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD",
        "CRM", "NFLX", "ADBE", "NOW", "CRWD", "ZS", "DDOG",
        "MDB", "NET", "SHOP", "UBER", "ABNB"
    ]

    cwh_detector = CupWithHandleDetector()
    db_detector = DoubleBottomDetector()
    index_df = fetch_index_data()

    # Track results: unfiltered vs quality-filtered
    results_unfiltered = {"success": 0, "failure": 0, "total": 0}
    results_filtered = {"success": 0, "failure": 0, "total": 0}

    min_gain_pct = 20
    max_loss_pct = 7
    outcome_window_days = 40

    logger.info(f"\nAnalyzing {len(test_symbols)} stocks...")
    logger.info(f"Success criteria: {min_gain_pct}% gain before {max_loss_pct}% loss in {outcome_window_days} days")

    for symbol in test_symbols:
        try:
            data = yf.download(symbol, period="3y", progress=False)
            if data.empty:
                continue
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            data.columns = [c.lower().replace(" ", "_") for c in data.columns]

            # Detect patterns
            patterns = cwh_detector.detect(symbol, data) + db_detector.detect(symbol, data)

            for pattern in patterns:
                # Evaluate outcome
                outcome = _evaluate_pattern_outcome(
                    pattern, data, min_gain_pct, max_loss_pct, outcome_window_days
                )

                if outcome == "pending":
                    continue

                # Unfiltered results
                results_unfiltered["total"] += 1
                if outcome == "success":
                    results_unfiltered["success"] += 1
                else:
                    results_unfiltered["failure"] += 1

                # Quality-filtered results
                try:
                    features = extract_all_features(
                        symbol=symbol,
                        stock_df=data,
                        index_df=index_df,
                        base_start_date=pattern.base_start_date,
                        base_end_date=pattern.base_end_date,
                        breakout_date=pattern.pivot_date,
                        pattern_metadata=pattern.metadata,
                    )

                    quality = compute_quality_score(
                        stock_df=data,
                        index_df=index_df,
                        features=features,
                        base_start_date=pattern.base_start_date,
                        base_end_date=pattern.base_end_date,
                    )

                    # Only count if passes quality filter
                    if quality.total_score >= 0.5:
                        results_filtered["total"] += 1
                        if outcome == "success":
                            results_filtered["success"] += 1
                        else:
                            results_filtered["failure"] += 1

                except Exception:
                    pass

        except Exception as e:
            continue

    # Print results
    logger.info("\n" + "-" * 50)
    logger.info("UNFILTERED PATTERNS:")
    if results_unfiltered["total"] > 0:
        rate = results_unfiltered["success"] / results_unfiltered["total"] * 100
        logger.info(f"  Total: {results_unfiltered['total']}")
        logger.info(f"  Success: {results_unfiltered['success']}")
        logger.info(f"  Failure: {results_unfiltered['failure']}")
        logger.info(f"  Success Rate: {rate:.1f}%")
    else:
        logger.info("  No patterns found")

    logger.info("\nQUALITY-FILTERED PATTERNS (score >= 0.5):")
    if results_filtered["total"] > 0:
        rate = results_filtered["success"] / results_filtered["total"] * 100
        logger.info(f"  Total: {results_filtered['total']}")
        logger.info(f"  Success: {results_filtered['success']}")
        logger.info(f"  Failure: {results_filtered['failure']}")
        logger.info(f"  Success Rate: {rate:.1f}%")

        # Calculate improvement
        if results_unfiltered["total"] > 0:
            unfiltered_rate = results_unfiltered["success"] / results_unfiltered["total"] * 100
            improvement = rate - unfiltered_rate
            logger.info(f"  Improvement: {improvement:+.1f}pp")
    else:
        logger.info("  No patterns passed quality filter")

    return results_unfiltered, results_filtered


def _evaluate_pattern_outcome(
    pattern, df: pd.DataFrame, min_gain_pct: float, max_loss_pct: float, window_days: int
) -> str:
    """Evaluate if a pattern led to a successful breakout."""
    try:
        pivot_date = datetime.strptime(pattern.base_end_date, "%Y-%m-%d")
        pivot_price = pattern.pivot_price

        df["date_parsed"] = pd.to_datetime(df["date"])
        post_pivot = df[df["date_parsed"] > pivot_date].head(window_days)

        if len(post_pivot) < 10:
            return "pending"

        success_price = pivot_price * (1 + min_gain_pct / 100)
        failure_price = pivot_price * (1 - max_loss_pct / 100)

        for _, row in post_pivot.iterrows():
            if row["low"] <= failure_price:
                return "failure"
            if row["high"] >= success_price:
                return "success"

        return "failure"

    except Exception:
        return "pending"


def spot_check_with_quality(n_samples: int = 5):
    """Spot check patterns with quality scores."""
    logger.info("\n" + "=" * 60)
    logger.info("SPOT CHECK WITH QUALITY SCORES")
    logger.info("=" * 60)

    test_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    index_df = fetch_index_data(period="5y")

    cwh_detector = CupWithHandleDetector()
    db_detector = DoubleBottomDetector()

    samples_shown = 0

    for symbol in test_symbols:
        if samples_shown >= n_samples:
            break

        try:
            # Use 5 years of data to ensure enough history for prior uptrend calculation
            data = yf.download(symbol, period="5y", progress=False)
            if data.empty:
                continue
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            data.columns = [c.lower().replace(" ", "_") for c in data.columns]

            patterns = cwh_detector.detect(symbol, data) + db_detector.detect(symbol, data)

            for pattern in patterns[:1]:  # Just first pattern per symbol
                if samples_shown >= n_samples:
                    break

                features = extract_all_features(
                    symbol=symbol,
                    stock_df=data,
                    index_df=index_df,
                    base_start_date=pattern.base_start_date,
                    base_end_date=pattern.base_end_date,
                    breakout_date=pattern.pivot_date,
                    pattern_metadata=pattern.metadata,
                )

                quality = compute_quality_score(
                    stock_df=data,
                    index_df=index_df,
                    features=features,
                    base_start_date=pattern.base_start_date,
                    base_end_date=pattern.base_end_date,
                )

                logger.info(f"\n{symbol} - {pattern.pattern_type.upper()}")
                logger.info(f"Period: {pattern.base_start_date} to {pattern.base_end_date}")
                logger.info(f"Pivot: ${pattern.pivot_price:.2f}")
                logger.info("")
                logger.info(get_quality_summary(quality))

                samples_shown += 1

        except Exception as e:
            continue


def main():
    logger.info("=" * 60)
    logger.info("PATTERN DETECTOR VALIDATION")
    logger.info("=" * 60)
    logger.info("\nTesting pattern detection accuracy and success rates")

    # 1. Validate against known patterns
    cwh_found, cwh_total = validate_cup_with_handle()
    db_found, db_total = validate_double_bottom()

    # 2. Measure success rates with/without quality filtering
    results_unfiltered, results_filtered = measure_success_rates()

    # 3. Spot check with quality scores
    spot_check_with_quality(n_samples=3)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nKnown Pattern Detection:")
    logger.info(f"  Cup-with-handle: {cwh_found}/{cwh_total} ({cwh_found/cwh_total*100:.0f}%)")
    logger.info(f"  Double-bottom:   {db_found}/{db_total} ({db_found/db_total*100:.0f}%)")

    total_found = cwh_found + db_found
    total = cwh_total + db_total
    logger.info(f"  Overall:         {total_found}/{total} ({total_found/total*100:.0f}%)")

    logger.info("\nSuccess Rate Impact of Quality Filtering:")
    if results_unfiltered["total"] > 0 and results_filtered["total"] > 0:
        unf_rate = results_unfiltered["success"] / results_unfiltered["total"] * 100
        filt_rate = results_filtered["success"] / results_filtered["total"] * 100
        reduction = (1 - results_filtered["total"] / results_unfiltered["total"]) * 100

        logger.info(f"  Unfiltered: {unf_rate:.1f}% success ({results_unfiltered['total']} patterns)")
        logger.info(f"  Filtered:   {filt_rate:.1f}% success ({results_filtered['total']} patterns)")
        logger.info(f"  Pattern reduction: {reduction:.0f}%")
        logger.info(f"  Success rate improvement: {filt_rate - unf_rate:+.1f}pp")


if __name__ == "__main__":
    main()
