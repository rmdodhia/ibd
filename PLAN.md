# IBD Breakout Scanner — Project Plan

## Overview

A Python-based stock screening system that detects IBD/CAN SLIM chart patterns, predicts breakout success using ML, and improves over time through continuous retraining. Runs locally, scans ~3,500 filtered NYSE+NASDAQ stocks.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     config.yaml                         │
│          (all thresholds, universe, parameters)          │
└──────────────────────┬──────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
│  Phase 1    │ │  Phase 2    │ │  Phase 3        │
│  Data       │ │  Labeler +  │ │  ML Training    │
│  Pipeline   │ │  Feature    │ │  Pipeline       │
│             │ │  Extractor  │ │  (reusable)     │
│ - yfinance  │ │             │ │                 │
│ - SQLite    │ │ - Breakout  │ │ - LightGBM     │
│ - OHLCV     │ │   detector  │ │ - 1D CNN       │
│ - Fundmntls │ │ - Feature   │ │ - Hybrid       │
│ - Daily     │ │   extract   │ │ - Walk-forward │
│   updates   │ │ - Streamlit │ │   validation   │
│             │ │   label UI  │ │ - Hyperparameter│
│             │ │             │ │   tuning       │
└──────┬──────┘ └──────┬──────┘ └───────┬─────────┘
       │               │               │
       ▼               ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                    SQLite Database                       │
│  tables: stocks, daily_prices, fundamentals,            │
│          detected_patterns, labels, predictions,         │
│          outcomes, model_runs                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Phase 4        │
              │  Live Scanner   │
              │                 │
              │ - On-demand CLI │
              │ - Prediction    │
              │   logger        │
              │ - Outcome       │
              │   tracker       │
              │ - Periodic      │
              │   retraining    │
              └─────────────────┘
```

---

## Phase 1 — Data Pipeline

**Goal:** Populate a local SQLite database with 10 years of daily price data and fundamentals for ~3,500 filtered stocks.

**Duration:** Weeks 1–2

### 1.1 Stock Universe

- Source: S&P 500 as primary, expand to NYSE+NASDAQ
- Pre-filter: price > $10, avg daily volume > 200K, market cap > $500M
- Store the universe list in `config.yaml` with a refresh script
- Also pull S&P 500 index data (for RS line calculation and market direction)

### 1.2 Data Sources

- **yfinance** for OHLCV (open, high, low, close, volume)
  - 10 years of daily bars per stock
  - Rate limit: ~2,000 requests/hour, use batching + sleep
  - Incremental daily updates after initial backfill
- **yfinance `ticker.info`** for fundamentals:
  - Market cap, shares outstanding, float
  - Institutional ownership %
  - Sector, industry
- **yfinance `ticker.quarterly_earnings`** for CAN SLIM fundamentals:
  - Quarterly EPS
  - Revenue
  - Derive: YoY EPS growth, revenue acceleration

### 1.3 Database Schema (SQLite)

```sql
-- Stock metadata
CREATE TABLE stocks (
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
CREATE TABLE daily_prices (
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
CREATE TABLE fundamentals (
    symbol TEXT,
    quarter_end DATE,
    eps REAL,
    revenue REAL,
    eps_yoy_growth REAL,
    revenue_yoy_growth REAL,
    PRIMARY KEY (symbol, quarter_end)
);

-- S&P 500 index (for RS line + market direction)
CREATE TABLE index_prices (
    symbol TEXT,  -- ^GSPC
    date DATE,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (symbol, date)
);
```

### 1.4 Scripts

- `scripts/backfill.py` — Initial data pull (10 years, all stocks). Handles rate limiting, retries, progress tracking. Resume-capable.
- `scripts/daily_update.py` — Pull latest day's data for all stocks. Run after market close.
- `scripts/refresh_universe.py` — Re-pull stock universe, add new stocks, flag delisted ones.

### 1.5 Deliverables

- [ ] SQLite database populated with 10yr data for ~3,500 stocks
- [ ] S&P 500 index data loaded
- [ ] Quarterly earnings/revenue loaded
- [ ] Daily update script working
- [ ] Data validation checks (no gaps, no nulls in critical fields)

---

## Phase 2 — Historical Labeler + Feature Extractor

**Goal:** Scan historical data to find breakout events, label them as success/failure, extract features, and provide a UI for human review.

**Duration:** Weeks 2–4

### 2.1 Breakout Detection (Historical)

Scan each stock's price history to find potential breakout events:

1. **Find consolidation bases:**
   - Identify periods where price trades in a range (high-to-low range < 35% of high) for 5+ weeks
   - Use a sliding window approach: for each day, look back 5–65 weeks for base-like behavior

2. **Find breakout attempts:**
   - Price closes above the consolidation range high (the "pivot") on above-average volume
   - Volume on breakout day > 1.4x 50-day average volume

3. **Label outcomes:**
   - **Success:** Stock gains ≥ 20% from pivot within 8 weeks, without first closing > 7% below pivot
   - **Failure:** Stock closes > 7% below pivot before gaining 20%, OR fails to gain 20% within 8 weeks
   - **Configurable thresholds in config.yaml**

### 2.2 Pattern Classification

For each detected breakout, classify the preceding base pattern:

**Cup with Handle:**
- Find peak (left lip) → decline 12–35% → rounded trough → recovery to within 5% of peak (right lip)
- Then small pullback (handle): 5–15% decline, 1–4 weeks, downward drift
- Pivot = high of handle
- Cup duration: 7–65 weeks
- Roundness check: bottom is not a sharp V (measure by checking price spends ≥3 weeks in bottom 1/3 of the cup)

**Double Bottom:**
- Peak → decline to first low → rally to mid-peak → decline to second low (within 5% of first, or slightly undercutting)
- Pivot = mid-peak price
- Duration: 7+ weeks
- Second low should ideally slightly undercut first (shakeout)

**Flat Base:**
- Price range from high to low ≤ 15%
- Duration ≥ 5 weeks
- Pivot = high of the base
- Tight weekly closes (low standard deviation)

**Ascending Base (future):**
- Three distinct pullbacks of 10–20%, each with a higher low
- Duration: 9–16 weeks
- Occurs during broader market pullback

**High Tight Flag (future):**
- Prior advance of ≥ 100% in 4–8 weeks
- Consolidation pullback of only 10–25%
- Very tight, low-volume flag

### 2.3 Feature Extraction

For each detected pattern, extract these features and store in the database:

**Pattern geometry features:**
- `base_depth_pct` — % decline from peak to trough
- `base_duration_weeks` — length of consolidation
- `base_symmetry` — ratio of left side to right side duration
- `handle_depth_pct` — handle decline as % of base depth (cup patterns)
- `handle_duration_weeks`
- `tightness_score` — std dev of weekly closes / average close within base
- `pivot_price` — the breakout price level
- `price_vs_50dma` — price at pivot relative to 50-day MA
- `price_vs_200dma` — price at pivot relative to 200-day MA
- `support_touches` — number of times price bounced near the low of the base
- `resistance_touches` — number of times price tested the pivot area
- `pattern_type` — categorical: cup_handle, double_bottom, flat_base, etc.

**Volume features:**
- `volume_trend_in_base` — slope of volume regression within base (declining = good)
- `breakout_volume_ratio` — breakout day volume / 50-day avg volume
- `up_down_volume_ratio` — ratio of volume on up days vs down days in base
- `volume_dry_up` — lowest volume week in base / average volume

**Relative strength features:**
- `rs_line_slope_4wk` — RS line trend over 4 weeks before pivot
- `rs_line_slope_12wk` — RS line trend over 12 weeks before pivot
- `rs_new_high` — boolean: did RS line hit new 52-week high at/before breakout?
- `rs_rank_percentile` — stock's RS percentile rank vs universe

**Fundamental features:**
- `eps_latest_yoy_growth` — most recent quarter EPS growth vs year-ago
- `eps_acceleration` — is EPS growth accelerating vs prior quarters?
- `revenue_latest_yoy_growth`
- `revenue_acceleration`
- `institutional_pct`
- `market_cap_log` — log of market cap (normalize scale)

**Market context features:**
- `sp500_above_200dma` — boolean: is S&P 500 above its 200-day MA? (market direction)
- `sp500_trend_4wk` — S&P 500 return over last 4 weeks
- `sector_relative_strength` — stock's sector performance vs S&P 500

### 2.4 Database Schema (additional tables)

```sql
-- Detected patterns (auto-labeled)
CREATE TABLE detected_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    pattern_type TEXT,          -- cup_handle, double_bottom, flat_base
    base_start_date DATE,
    base_end_date DATE,
    pivot_date DATE,
    pivot_price REAL,
    outcome TEXT,               -- success, failure, pending
    outcome_return_pct REAL,    -- actual return after breakout
    outcome_max_gain_pct REAL,  -- max gain before any reversal
    outcome_max_loss_pct REAL,  -- max drawdown after breakout
    auto_label TEXT,            -- auto-assigned label
    human_label TEXT,           -- NULL until reviewed, overrides auto_label
    reviewed BOOLEAN DEFAULT 0,
    created_at TIMESTAMP
);

-- Extracted features for each pattern
CREATE TABLE pattern_features (
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
    FOREIGN KEY (pattern_id) REFERENCES detected_patterns(id)
);

-- Live predictions (Phase 4)
CREATE TABLE predictions (
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
CREATE TABLE model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TIMESTAMP,
    model_type TEXT,            -- lightgbm, cnn, hybrid
    model_version TEXT,
    train_start DATE,
    train_end DATE,
    test_start DATE,
    test_end DATE,
    precision REAL,
    recall REAL,
    f1_score REAL,
    accuracy REAL,
    n_train_samples INTEGER,
    n_test_samples INTEGER,
    hyperparameters TEXT,       -- JSON blob
    feature_importances TEXT,   -- JSON blob
    model_path TEXT             -- path to saved model file
);
```

### 2.5 Labeling UI (Streamlit)

A local web app for reviewing and correcting auto-labels:

- **Main view:** Paginated list of detected patterns, sortable/filterable by symbol, pattern type, outcome, reviewed status
- **Detail view:** Interactive price chart (plotly) showing:
  - The base pattern highlighted
  - Pivot point marked
  - Volume bars below
  - 50-day and 200-day moving averages
  - RS line overlay
  - The post-breakout price action (outcome)
- **Label controls:**
  - Confirm auto-label
  - Override: change pattern type, mark as "not a pattern", adjust base start/end dates
  - Flag as ambiguous
- **Progress tracker:** X of Y patterns reviewed

### 2.6 Deliverables

- [ ] Historical breakout detector working
- [ ] Pattern classifier working for: cup_handle, double_bottom, flat_base
- [ ] Feature extractor populating pattern_features table
- [ ] Streamlit labeling UI running locally
- [ ] At least 500 patterns reviewed and labeled (first pass)

---

## Phase 3 — ML Training Pipeline (Reusable)

**Goal:** A single training pipeline that trains, evaluates, and saves models. Re-runnable anytime with new data.

**Duration:** Weeks 4–6 (LightGBM), Weeks 10–14 (CNN), Weeks 14–18 (Hybrid)

### 3.1 Training Pipeline Architecture

```
training_pipeline.py
├── load_data()           — Query labeled patterns + features from SQLite
├── prepare_splits()      — Walk-forward time-series splits
├── train_lightgbm()      — Train + tune LightGBM
├── train_cnn()           — Train + tune 1D CNN
├── train_hybrid()        — Train + tune hybrid model
├── evaluate()            — Precision, recall, F1, hit rate, profit factor
├── compare_models()      — Side-by-side comparison of all trained models
├── save_model()          — Save model + metadata to model_runs table
└── generate_report()     — HTML report of training results
```

### 3.2 Walk-Forward Validation

```
Split 1: Train [2016–2019] → Test [2020]
Split 2: Train [2016–2020] → Test [2021]
Split 3: Train [2016–2021] → Test [2022]
Split 4: Train [2016–2022] → Test [2023]
Split 5: Train [2016–2023] → Test [2024]
Split 6: Train [2016–2024] → Test [2025]

Each split:
  → 80/20 train/validation within training window (for hyperparam tuning)
  → Test set is untouched until final evaluation
  → Record per-split metrics to detect regime sensitivity
```

### 3.3 Stage A — LightGBM (Weeks 4–6)

**Input:** Tabular features from `pattern_features` table
**Output:** P(successful breakout)

- Hyperparameter search via Optuna:
  - `num_leaves`: [20, 150]
  - `learning_rate`: [0.01, 0.3]
  - `min_child_samples`: [5, 50]
  - `reg_alpha`: [0, 10]
  - `reg_lambda`: [0, 10]
  - `feature_fraction`: [0.5, 1.0]
- Class imbalance handling: `scale_pos_weight` or SMOTE
- Feature importance: SHAP values for interpretability

### 3.4 Stage B — 1D CNN (Weeks 10–14)

**Prerequisite:** ≥15,000 labeled examples

**Input:** Raw daily price series, 200 trading days before pivot
- Channels: close (normalized), volume (normalized), RS line, 50DMA ratio, 200DMA ratio
- Shape: (200 days × 5 channels)

**Architecture:**
```
Input (200 × 5)
→ Conv1D(filters=64, kernel=7, stride=1) + BatchNorm + ReLU
→ Conv1D(filters=64, kernel=5) + BatchNorm + ReLU
→ MaxPool1D(2)
→ Conv1D(filters=128, kernel=5) + BatchNorm + ReLU
→ Conv1D(filters=128, kernel=3) + BatchNorm + ReLU
→ GlobalAveragePooling1D
→ Dense(64) + Dropout(0.3)
→ Dense(1, sigmoid)
```

- Framework: PyTorch
- Optimizer: AdamW with cosine annealing LR schedule
- Early stopping on validation loss

### 3.5 Stage C — Hybrid (Weeks 14–18)

**Input:** CNN learned features + tabular features

```
Raw price series → CNN backbone (freeze or fine-tune) → 64-dim feature vector
                                                              ↓
Tabular features (from pattern_features) → Dense(32) ──→ Concatenate → Dense(64) → Dense(1, sigmoid)
```

Alternative: use CNN features as additional columns in LightGBM (simpler, often works just as well).

### 3.6 Evaluation Metrics

For each model, track:
- **Precision:** Of stocks flagged as breakouts, what % actually succeeded?
- **Recall:** Of actual breakouts, what % did we catch?
- **F1 Score:** Harmonic mean of precision and recall
- **Hit Rate by Confidence Tier:** Break predictions into quartiles by confidence; top quartile should have highest hit rate
- **Profit Factor:** (Sum of gains from successful predictions) / (Sum of losses from failed predictions) — simulates real trading
- **Regime Analysis:** Performance broken down by bull/bear/choppy market conditions

### 3.7 Retraining Process

The same pipeline supports both initial training and periodic retraining:

```bash
# Initial training
python -m scanner.train --model lightgbm --full-history

# Retrain with new data (same command, picks up new labels/outcomes)
python -m scanner.train --model lightgbm --full-history

# Compare new model vs current production model
python -m scanner.train --model lightgbm --compare-to v1.2

# Promote new model to production
python -m scanner.promote --model-version v1.3
```

### 3.8 Deliverables

- [ ] LightGBM v1 trained, evaluated, baseline metrics recorded
- [ ] Training pipeline re-runnable with one command
- [ ] SHAP-based feature importance report
- [ ] Walk-forward results showing per-year performance
- [ ] Model saved and versioned

---

## Phase 4 — Live Scanner + Prediction Logger

**Goal:** Scan current stock data, flag candidates, log predictions, track outcomes.

**Duration:** Weeks 6+ (ongoing)

### 4.1 Scanner CLI

```bash
# Run a full scan of the universe
python -m scanner.scan

# Scan a specific stock
python -m scanner.scan --symbol AAPL

# Scan with minimum confidence threshold
python -m scanner.scan --min-confidence 0.7

# Scan and output results as JSON
python -m scanner.scan --output json
```

**Scanner output per candidate:**
- Symbol, current price, pivot price, distance from pivot (%)
- Pattern type, base start/end dates
- Confidence score from ML model
- Top 3 contributing features (from SHAP)
- RS line status (new high? rising?)
- Fundamental snapshot (latest EPS growth, revenue growth)
- Volume trend assessment

### 4.2 Prediction Logging

Every scan result above the confidence threshold gets logged to the `predictions` table:
- Symbol, date, pattern type, pivot price
- Model version, confidence score
- Outcome: initially "pending"

### 4.3 Outcome Tracker

```bash
# Check outcomes of pending predictions
python -m scanner.check_outcomes

# This runs daily (or on-demand) and:
# 1. For each pending prediction, pull latest price data
# 2. Check if success criteria met (≥20% gain within 8 weeks)
# 3. Check if failure criteria met (>7% below pivot)
# 4. Update predictions table with outcome
# 5. Optionally trigger retraining if enough new outcomes accumulated
```

### 4.4 Drift Monitor

```bash
# Compare live performance to backtest expectations
python -m scanner.drift_report

# Shows:
# - Rolling 30-day hit rate vs backtest hit rate
# - Confidence calibration (are 80% confidence picks actually hitting 80%?)
# - Feature distribution shift (are current patterns different from training data?)
# - Recommendation: retrain or hold
```

### 4.5 Deliverables

- [ ] CLI scanner producing ranked candidate list
- [ ] Predictions logged to database
- [ ] Outcome tracker updating resolved predictions
- [ ] Drift monitoring report
- [ ] Retraining triggered when drift detected or N new outcomes accumulated

---

## Global Configuration (config.yaml)

All tunable parameters in one place:

```yaml
# ── Universe ──────────────────────────────
universe:
  source: "sp500"                    # sp500, nyse_nasdaq, custom
  min_price: 10
  min_avg_volume: 200000
  min_market_cap: 500000000          # $500M

# ── Data ──────────────────────────────────
data:
  history_years: 10
  database_path: "data/ibd_scanner.db"
  yfinance_batch_size: 50
  yfinance_sleep_seconds: 1

# ── Breakout Definition ───────────────────
breakout:
  min_gain_pct: 20                   # minimum % gain to count as success
  max_loss_pct: 7                    # max % decline before counted as failure
  outcome_window_weeks: 8            # how long to wait for outcome
  min_breakout_volume_ratio: 1.4     # breakout volume vs 50-day avg

# ── Pattern Detection ────────────────────
patterns:
  sensitivity: 0.5                   # 0.0 (strict) to 1.0 (loose)

  cup_with_handle:
    enabled: true
    min_depth_pct: 12
    max_depth_pct: 35
    min_duration_weeks: 7
    max_duration_weeks: 65
    handle_max_depth_pct: 15
    handle_min_weeks: 1
    handle_max_weeks: 4
    roundness_min_weeks_in_bottom: 3

  double_bottom:
    enabled: true
    min_depth_pct: 12
    max_depth_pct: 35
    min_duration_weeks: 7
    second_low_tolerance_pct: 5      # how close second low must be to first

  flat_base:
    enabled: true
    max_depth_pct: 15
    min_duration_weeks: 5

  ascending_base:
    enabled: false                   # enable later
    pullback_min_pct: 10
    pullback_max_pct: 20
    min_pullbacks: 3

  high_tight_flag:
    enabled: false                   # enable later
    min_prior_gain_pct: 100
    max_flag_depth_pct: 25

# ── Feature Extraction ───────────────────
features:
  rs_benchmark: "^GSPC"             # S&P 500 for relative strength
  moving_averages: [50, 200]
  volume_avg_period: 50

# ── ML Training ──────────────────────────
training:
  model_type: "lightgbm"            # lightgbm, cnn, hybrid
  walk_forward_test_years: 1
  min_train_samples: 500
  class_balance_method: "scale_pos_weight"  # scale_pos_weight, smote, none
  hyperparam_trials: 100            # Optuna trials
  random_seed: 42

  lightgbm:
    num_leaves_range: [20, 150]
    learning_rate_range: [0.01, 0.3]
    min_child_samples_range: [5, 50]

  cnn:
    lookback_days: 200
    channels: ["close_norm", "volume_norm", "rs_line", "ma50_ratio", "ma200_ratio"]
    epochs: 100
    batch_size: 64
    early_stopping_patience: 10

# ── Scanner ──────────────────────────────
scanner:
  min_confidence: 0.6
  max_results: 50
  output_format: "table"            # table, json, csv

# ── Retraining ───────────────────────────
retraining:
  trigger_new_outcomes: 100         # retrain after N new outcomes
  min_retrain_interval_days: 30
  drift_threshold: 0.10             # retrain if hit rate drops > 10% vs backtest
```

---

## File Structure

```
ibd-scanner/
├── PLAN.md                         ← this file
├── CLAUDE.md                       ← context for Claude Code CLI
├── README.md                       ← setup + usage instructions
├── config.yaml                     ← all tunable parameters
├── requirements.txt                ← Python dependencies
├── data/
│   └── ibd_scanner.db             ← SQLite database (created by backfill)
├── scanner/
│   ├── __init__.py
│   ├── db.py                      ← database connection + helpers
│   ├── data_pipeline.py           ← yfinance pulls, daily updates
│   ├── universe.py                ← stock universe management
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── base_detector.py       ← shared: peak/trough detection, smoothing
│   │   ├── cup_with_handle.py
│   │   ├── double_bottom.py
│   │   ├── flat_base.py
│   │   ├── ascending_base.py      ← future
│   │   └── high_tight_flag.py     ← future
│   ├── features/
│   │   ├── __init__.py
│   │   ├── pattern_features.py    ← geometry features
│   │   ├── volume_features.py     ← volume-based features
│   │   ├── rs_features.py         ← relative strength features
│   │   ├── fundamental_features.py ← earnings, revenue, etc.
│   │   └── market_features.py     ← S&P 500, sector context
│   ├── models/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py   ← main train/evaluate/save pipeline
│   │   ├── lightgbm_model.py      ← LightGBM specifics
│   │   ├── cnn_model.py           ← 1D CNN (PyTorch)
│   │   ├── hybrid_model.py        ← CNN + tabular hybrid
│   │   └── saved/                 ← serialized model files
│   ├── labeler.py                 ← historical breakout detection + labeling
│   ├── scan.py                    ← live scanner entry point
│   ├── train.py                   ← training CLI entry point
│   ├── check_outcomes.py          ← outcome resolution
│   └── drift_report.py           ← drift monitoring
├── labeler/
│   ├── app.py                     ← Streamlit labeling UI
│   └── components/                ← UI components
├── scripts/
│   ├── backfill.py                ← initial data backfill
│   ├── daily_update.py            ← daily data pull
│   └── refresh_universe.py        ← update stock universe
├── tests/
│   ├── test_patterns.py
│   ├── test_features.py
│   ├── test_labeler.py
│   └── test_pipeline.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_pattern_validation.ipynb
    └── 03_model_analysis.ipynb
```

---

## Task Checklist

### Phase 1 — Data Pipeline (Weeks 1–2)
- [ ] Set up project structure and dependencies
- [ ] Implement `db.py` — SQLite connection, table creation
- [ ] Implement `universe.py` — pull S&P 500 symbols, filtering
- [ ] Implement `data_pipeline.py` — yfinance OHLCV + fundamentals pull
- [ ] Implement `scripts/backfill.py` — full historical backfill with progress
- [ ] Implement `scripts/daily_update.py` — incremental daily update
- [ ] Load S&P 500 index data for RS calculations
- [ ] Data validation: check for gaps, nulls, sanity checks
- [ ] Test with 10 stocks end-to-end before full backfill

### Phase 2 — Labeler + Features (Weeks 2–4)
- [ ] Implement `patterns/base_detector.py` — peak/trough detection, smoothing
- [ ] Implement `patterns/cup_with_handle.py`
- [ ] Implement `patterns/double_bottom.py`
- [ ] Implement `patterns/flat_base.py`
- [ ] Implement `labeler.py` — historical breakout detection + auto-labeling
- [ ] Implement all feature extractors (pattern, volume, RS, fundamental, market)
- [ ] Implement `labeler/app.py` — Streamlit UI for label review
- [ ] Run labeler on full dataset, generate initial labels
- [ ] Review and correct at least 500 labels

### Phase 3 — ML Training (Weeks 4–6, then 10–18)
- [ ] Implement `training_pipeline.py` — data loading, splits, evaluation
- [ ] Implement `lightgbm_model.py` — training, Optuna tuning
- [ ] Train LightGBM v1, record baseline metrics
- [ ] SHAP analysis + feature importance report
- [ ] Walk-forward results across all time periods
- [ ] (Later) Implement `cnn_model.py` when data volume sufficient
- [ ] (Later) Implement `hybrid_model.py`
- [ ] (Later) Compare all models side-by-side

### Phase 4 — Live Scanner (Weeks 6+)
- [ ] Implement `scan.py` — CLI scanner using trained model
- [ ] Implement prediction logging
- [ ] Implement `check_outcomes.py` — outcome tracker
- [ ] Implement `drift_report.py` — model monitoring
- [ ] Implement periodic retraining trigger
- [ ] First live scan + predictions logged
