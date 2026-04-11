"""Database connection, schema creation, and helper functions."""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager

from scanner.config import get

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Stock metadata
CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap REAL,
    shares_outstanding REAL,
    float_shares REAL,
    institutional_pct REAL,
    last_updated DATE
);

-- Daily OHLCV
CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);

-- Quarterly fundamentals
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT,
    quarter_end DATE,
    eps REAL,
    revenue REAL,
    eps_yoy_growth REAL,
    revenue_yoy_growth REAL,
    PRIMARY KEY (symbol, quarter_end)
);

-- S&P 500 index data
CREATE TABLE IF NOT EXISTS index_prices (
    symbol TEXT,
    date DATE,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);

-- Detected patterns (auto + human labeled)
CREATE TABLE IF NOT EXISTS detected_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    pattern_type TEXT,
    base_start_date DATE,
    base_end_date DATE,
    pivot_date DATE,
    pivot_price REAL,
    -- Legacy outcome column (for backward compatibility)
    outcome TEXT,
    outcome_return_pct REAL,
    outcome_max_gain_pct REAL,
    outcome_max_loss_pct REAL,
    -- Multi-label outcomes (for experimentation)
    outcome_asym_20_7 TEXT,      -- +20%/-7% (original IBD rules)
    outcome_asym_15_10 TEXT,     -- +15%/-10% (less extreme)
    outcome_sym_10 TEXT,         -- +10%/-10% (symmetric)
    return_asym_20_7 REAL,       -- Return % for each strategy
    return_asym_15_10 REAL,
    return_sym_10 REAL,
    auto_label TEXT,
    human_label TEXT,
    reviewed BOOLEAN DEFAULT 0,
    -- Pattern confidence and metadata
    confidence REAL,             -- Detector confidence score (0-1)
    metadata TEXT,               -- JSON-encoded pattern metadata
    pattern_type_override TEXT,  -- User's correction of pattern type
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted features per pattern
CREATE TABLE IF NOT EXISTS pattern_features (
    pattern_id INTEGER PRIMARY KEY,
    base_depth_pct REAL,
    base_duration_weeks REAL,
    base_symmetry REAL,
    handle_depth_pct REAL,
    tightness_score REAL,
    breakout_volume_ratio REAL,
    volume_trend_in_base REAL,
    up_down_volume_ratio REAL,
    rs_line_slope_4wk REAL,
    rs_line_slope_12wk REAL,
    rs_new_high BOOLEAN,
    rs_rank_percentile REAL,
    eps_latest_yoy_growth REAL,
    eps_acceleration REAL,
    revenue_latest_yoy_growth REAL,
    institutional_pct REAL,
    market_cap_log REAL,
    sp500_above_200dma BOOLEAN,
    sp500_trend_4wk REAL,
    price_vs_50dma REAL,
    price_vs_200dma REAL,
    -- Quality scores
    quality_score REAL,
    technical_score REAL,
    fundamental_score REAL,
    market_score REAL,
    prior_uptrend_pct REAL,
    FOREIGN KEY (pattern_id) REFERENCES detected_patterns(id)
);

-- Live predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER,
    symbol TEXT,
    prediction_date DATE,
    model_version TEXT,
    confidence_score REAL,
    predicted_outcome TEXT,
    actual_outcome TEXT,
    actual_return_pct REAL,
    resolved_date DATE,
    FOREIGN KEY (pattern_id) REFERENCES detected_patterns(id)
);

-- Model training runs
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type TEXT,
    model_version TEXT,
    train_start DATE,
    train_end DATE,
    test_start DATE,
    test_end DATE,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    accuracy REAL,
    n_train_samples INTEGER,
    n_test_samples INTEGER,
    hyperparameters TEXT,
    feature_importances TEXT,
    model_path TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol);
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_detected_patterns_symbol ON detected_patterns(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
"""


def get_db_path() -> Path:
    """Get database path from config."""
    db_path = Path(get("data.database_path", "data/ibd_scanner.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with optimized settings.

    Returns:
        sqlite3.Connection with WAL mode and foreign keys enabled.
    """
    conn = sqlite3.connect(get_db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_cursor():
    """Context manager for database cursor with auto-commit.

    Yields:
        sqlite3.Cursor

    Example:
        >>> with get_cursor() as cur:
        ...     cur.execute("SELECT * FROM stocks")
        ...     rows = cur.fetchall()
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables and indexes if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        logger.info("Database initialized at %s", get_db_path())
    finally:
        conn.close()

    # Run migrations for existing databases
    migrate_db()


def migrate_db() -> None:
    """Add new columns to existing database tables.

    This handles upgrading existing databases to new schema versions.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Check if multi-label columns exist in detected_patterns
        cur.execute("PRAGMA table_info(detected_patterns)")
        columns = {row[1] for row in cur.fetchall()}

        # Add multi-label outcome columns if missing
        new_columns = [
            ("outcome_asym_20_7", "TEXT"),
            ("outcome_asym_15_10", "TEXT"),
            ("outcome_sym_10", "TEXT"),
            ("return_asym_20_7", "REAL"),
            ("return_asym_15_10", "REAL"),
            ("return_sym_10", "REAL"),
            ("confidence", "REAL"),
            ("metadata", "TEXT"),
            ("pattern_type_override", "TEXT"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    cur.execute(f"ALTER TABLE detected_patterns ADD COLUMN {col_name} {col_type}")
                    logger.info("Added column %s to detected_patterns", col_name)
                except Exception as e:
                    logger.debug("Column %s may already exist: %s", col_name, e)

        conn.commit()
    finally:
        conn.close()
