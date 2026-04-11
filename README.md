# IBD Breakout Scanner

A machine learning system that learns to identify chart patterns that precede successful stock breakouts. Uses a hybrid CNN + tabular model to analyze price action shapes and fundamentals.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Pipeline](#data-pipeline)
4. [Training the Model](#training-the-model)
5. [Running the Scanner](#running-the-scanner)
6. [Improving Model Accuracy](#improving-model-accuracy)
7. [Monitoring and Retraining](#monitoring-and-retraining)
8. [Performance Optimizations](#performance-optimizations)
9. [Configuration](#configuration)
10. [Project Structure](#project-structure)

---

## Overview

The scanner uses a two-stage approach:

1. **Breakout Detection**: Finds all historical breakout attempts (price closes above N-day high on above-average volume)
2. **Shape Learning**: A CNN learns what price patterns precede successful breakouts directly from data

This differs from traditional rule-based pattern detection. Instead of manually coding "cup-with-handle" rules, the model learns which shapes work from historical outcomes.

### Architecture

```
Pre-breakout price series (200 days × 5 channels)
         ↓
    CNN Backbone → 128-dim shape embedding
         ↓
    Concatenate with tabular features (21 features)
         ↓
    Dense layers → P(successful breakout)
```

**Channels**: normalized close, volume ratio, RS line, MA50 ratio, MA200 ratio

**Tabular features**: EPS growth, revenue growth, RS rank, market regime, pattern geometry, etc.

---

## Installation

### Prerequisites

- Python 3.11+
- ~2GB disk space for database
- ~4GB RAM for training

### Setup

```bash
# Clone/download project
cd ibd-scanner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Data Pipeline

### Step 1: Initial Data Backfill

Pull 10 years of historical data for S&P 500 stocks. This takes 2-4 hours on first run.

```bash
# Full backfill (recommended)
python scripts/backfill.py

# Test with 10 stocks first (optional)
python scripts/backfill.py --test 10
```

**What it does:**
- Fetches S&P 500 constituents from Wikipedia
- Filters by market cap (>$500M), price (>$10), volume (>200K)
- Downloads 10 years of daily OHLCV data via yfinance
- Fetches quarterly fundamentals (EPS, revenue)
- Downloads S&P 500 index data for relative strength calculations

**Resume capability**: If interrupted, re-run the same command. It skips symbols that already have data.

### Step 2: Daily Updates

Run after market close (4 PM ET or later) to keep data current.

```bash
python scripts/daily_update.py

# Fetch more days if you missed updates
python scripts/daily_update.py --days 10
```

### Step 3: Refresh Universe (Quarterly)

Update the stock universe when S&P 500 constituents change.

```bash
python scripts/refresh_universe.py
```

### Verify Data

```bash
python -c "
from scanner.db import get_connection
conn = get_connection()
print('Stocks:', conn.execute('SELECT COUNT(*) FROM stocks').fetchone()[0])
print('Price rows:', conn.execute('SELECT COUNT(*) FROM daily_prices').fetchone()[0])
print('Date range:', conn.execute('SELECT MIN(date), MAX(date) FROM daily_prices').fetchone())
"
```

---

## Training the Model

### Step 1: Run Historical Labeler

Scan all historical data to find breakouts and label their outcomes.

```bash
# Label all stocks (takes 30-60 minutes)
python -m scanner.labeler

# Label a single stock (for testing)
python -m scanner.labeler --symbol AAPL

# Re-label stocks (overwrites existing labels)
python -m scanner.labeler --force
```

**What it does:**
- Finds all breakout attempts in historical data
- Labels outcomes: success (≥20% gain in 8 weeks) or failure (>7% loss first)
- Classifies pattern type (cup-with-handle, double-bottom, flat-base, or unclassified)
- Extracts 21 tabular features per breakout

**Expected output**: 20,000-40,000 labeled breakouts across 10 years of S&P 500 data.

### Step 2: Train the Model

```bash
# Train hybrid CNN + tabular model (recommended)
python -m scanner.train --model hybrid

# Assign a version name
python -m scanner.train --model hybrid --version v1.0

# Save to custom path
python -m scanner.train --model hybrid --save-path models/my_model.pt
```

**What it does:**
- Prepares 200-day price series tensors (N samples × 200 days × 5 channels)
- Creates walk-forward validation splits with 8-week embargo (prevents data leakage)
- Trains CNN + tabular hybrid model with early stopping
- Reports precision, recall, F1, AUC, and profit factor per split
- Saves final model trained on all data

**Training output:**
```
TRAINING RESULTS
============================================================
Version: v20240115_143022
Model saved to: models/breakout_predictor.pt

Aggregate Metrics (across walk-forward splits):
  Precision: 0.623 ± 0.045
  Recall:    0.571 ± 0.062
  F1 Score:  0.595 ± 0.041
  AUC:       0.684 ± 0.038
  Profit Factor: 1.85

Evaluated on 6 splits, 8432 total samples
============================================================
```

### Understanding the Metrics

| Metric | What it means | Target |
|--------|--------------|--------|
| **Precision** | % of predicted breakouts that succeed | >60% |
| **Recall** | % of actual successes we catch | >50% |
| **F1 Score** | Balance of precision and recall | >55% |
| **Profit Factor** | Wins / Losses (assuming equal position sizes) | >1.5 |

---

## Running the Scanner

### Daily Scan

```bash
# Scan full universe
python -m scanner.scan

# Scan single stock
python -m scanner.scan --symbol NVDA

# Filter by confidence
python -m scanner.scan --min-confidence 0.7

# Output as JSON or CSV
python -m scanner.scan --output json
python -m scanner.scan --output csv > candidates.csv

# Save predictions to database for tracking
python -m scanner.scan --save
```

**Output:**
```
================================================================================
BREAKOUT CANDIDATES
================================================================================

Symbol   Conf   Pattern            Price      From High    Vol
--------------------------------------------------------------------------------
NVDA     82.3%  cup_with_handle    $875.50        2.1%    1.8x
META     76.1%  flat_base          $485.20        1.5%    1.4x
AMZN     71.5%  double_bottom      $178.30        3.2%    1.6x
...
--------------------------------------------------------------------------------
Total: 12 candidates
```

### Check Prediction Outcomes

After 8 weeks, check how predictions resolved:

```bash
python -m scanner.check_outcomes
```

This updates the database with actual outcomes (success/failure) for predictions that have resolved.

---

## Improving Model Accuracy

The model learns from labeled data. To improve accuracy, you can correct mislabeled breakouts (false positives and false negatives).

### Understanding Errors

| Error Type | What happened | How to fix |
|------------|---------------|------------|
| **False Positive** | Model predicted success, but stock failed | Review if breakout criteria were valid |
| **False Negative** | Model predicted failure, but stock succeeded | Ensure pattern was properly detected |

### Step 1: Review Predictions in Database

```bash
python -c "
import pandas as pd
from scanner.db import get_connection

conn = get_connection()

# False positives: predicted success but failed
false_positives = pd.read_sql_query('''
    SELECT p.symbol, p.prediction_date, p.confidence_score,
           p.actual_outcome, p.actual_return_pct
    FROM predictions p
    WHERE p.confidence_score >= 0.6
      AND p.actual_outcome = 'failure'
    ORDER BY p.prediction_date DESC
    LIMIT 20
''', conn)
print('FALSE POSITIVES (high confidence failures):')
print(false_positives.to_string())

# False negatives: low confidence but succeeded
false_negatives = pd.read_sql_query('''
    SELECT p.symbol, p.prediction_date, p.confidence_score,
           p.actual_outcome, p.actual_return_pct
    FROM predictions p
    WHERE p.confidence_score < 0.5
      AND p.actual_outcome = 'success'
    ORDER BY p.prediction_date DESC
    LIMIT 20
''', conn)
print('\nFALSE NEGATIVES (low confidence successes):')
print(false_negatives.to_string())
"
```

### Step 2: Launch Labeling UI

Review and correct pattern labels visually:

```bash
streamlit run labeler/app.py
# Opens browser at http://localhost:8501
```

In the UI you can:
- View the price chart for each detected pattern
- Correct the pattern type classification
- Mark patterns as invalid (exclude from training)
- Add notes for ambiguous cases

### Step 3: Correct Labels in Database

For bulk corrections, you can update labels directly:

```bash
python -c "
from scanner.db import get_cursor

with get_cursor() as cur:
    # Mark a specific pattern as human-reviewed
    cur.execute('''
        UPDATE detected_patterns
        SET human_label = 'invalid', reviewed = 1
        WHERE id = 12345
    ''')

    # Correct pattern type
    cur.execute('''
        UPDATE detected_patterns
        SET pattern_type = 'cup_with_handle', reviewed = 1
        WHERE id = 12346
    ''')
"
```

### Step 4: Retrain with Corrected Labels

After corrections, retrain the model:

```bash
# Clear and re-run labeler to pick up corrections
python -m scanner.labeler --force

# Train new model version
python -m scanner.train --model hybrid --version v1.1
```

### Label Correction Best Practices

1. **Focus on high-confidence errors first** - False positives with >70% confidence are the most impactful to fix

2. **Look for systematic errors** - If the model consistently fails on a specific pattern type, investigate the detection rules

3. **Check the breakout criteria** - Some "breakouts" may be invalid (e.g., earnings gaps, low volume)

4. **Verify the outcome window** - A stock that dropped 7% on day 1 but recovered to +25% is labeled "failure" by default

5. **Consider market context** - Breakouts during market corrections have lower success rates

### Analyzing Feature Importance

To understand what the model learned:

```bash
python -c "
import torch
from scanner.models import load_model
from scanner.models.data_prep import get_feature_names

model, metadata = load_model('models/breakout_predictor.pt')
print('Model version:', metadata.get('version'))
print('Training samples:', metadata.get('n_samples'))
print('Metrics:', metadata.get('metrics'))
print('\nTabular features:', get_feature_names())
"
```

---

## Monitoring and Retraining

### Drift Report

Check if live performance differs from backtest:

```bash
python -m scanner.drift_report
```

**Output:**
```
============================================================
MODEL DRIFT REPORT
Generated: 2024-01-15
============================================================

BACKTEST METRICS (from training)
----------------------------------------
  Model version: v1.0
  Precision: 0.623
  Recall:    0.571
  F1 Score:  0.595

LIVE PERFORMANCE (last 30 days)
----------------------------------------
  Predictions: 45
  Success rate: 58.2%
  Precision: 0.612
  Recall:    0.545
  F1 Score:  0.576
  Avg Return: +8.3%
  Win Rate: 62.2%

RECOMMENDATION
----------------------------------------
  OK: No drift detected
============================================================
```

### When to Retrain

The system recommends retraining when:

1. **Drift detected**: Live precision/F1 drops >10% vs backtest
2. **New data available**: 100+ new resolved outcomes
3. **Market regime change**: Extended bear/bull market

### Retraining Workflow

```bash
# 1. Update data
python scripts/daily_update.py

# 2. Check outcomes for pending predictions
python -m scanner.check_outcomes

# 3. Run drift report
python -m scanner.drift_report

# 4. If retraining needed, re-run labeler and train
python -m scanner.labeler --force
python -m scanner.train --model hybrid --version v1.2

# 5. Verify improvement
python -m scanner.drift_report
```

---

## Performance Optimizations

The scanner includes several optimizations for faster training and processing.

### GPU Acceleration

Training automatically uses GPU if available:

- **NVIDIA CUDA**: Full support with mixed precision (AMP) for ~2x speedup
- **Apple MPS (Metal)**: Supported on M1/M2 Macs
- **CPU fallback**: Works on any system, just slower

```bash
# Check what device will be used
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple MPS (Metal)')
else:
    print('CPU only')
"
```

### Mixed Precision Training (AMP)

On NVIDIA GPUs, mixed precision training is enabled by default:
- Uses FP16 for forward/backward pass (faster, less memory)
- Uses FP32 for weight updates (maintains accuracy)
- ~2x speedup on modern GPUs (RTX 20xx and newer)

### Parallel Processing

**Labeling** uses multiprocessing to process symbols in parallel:

```bash
# Auto-detect optimal workers (default)
python -m scanner.labeler

# Specify number of workers
python -m scanner.labeler --workers 8

# Single-threaded (for debugging)
python -m scanner.labeler --workers 1
```

**Data loading** uses parallel workers:
- Auto-detects optimal worker count based on CPU cores
- Uses pinned memory for faster GPU transfer
- Persistent workers to avoid spawn overhead

**yfinance downloads** use threaded batch downloads.

### Memory Optimization

For large datasets or limited memory:

```yaml
# config.yaml - reduce batch size
training:
  cnn:
    batch_size: 32    # Default: 64, reduce if OOM
```

```bash
# Clear GPU cache between operations
python -c "import torch; torch.cuda.empty_cache()"
```

### Training Speed Benchmarks

Approximate training times (30,000 samples, 100 epochs):

| Hardware | Time | Notes |
|----------|------|-------|
| NVIDIA RTX 4090 | ~5 min | With AMP |
| NVIDIA RTX 3080 | ~10 min | With AMP |
| Apple M2 Pro | ~20 min | MPS backend |
| CPU (8 cores) | ~60 min | Parallel data loading helps |

### Optimizing for Your Hardware

**High-end GPU (24GB+ VRAM)**:
```yaml
training:
  cnn:
    batch_size: 128   # Larger batches = faster training
```

**Mid-range GPU (8-12GB VRAM)**:
```yaml
training:
  cnn:
    batch_size: 64    # Default
```

**Low VRAM or CPU**:
```yaml
training:
  cnn:
    batch_size: 32    # Smaller batches
    epochs: 50        # Fewer epochs, rely on early stopping
```

---

## Configuration

All parameters are in `config.yaml`. Key settings:

### Breakout Definition

```yaml
breakout:
  min_gain_pct: 20          # Success threshold
  max_loss_pct: 7           # Failure threshold
  outcome_window_weeks: 8   # How long to track
  min_breakout_volume_ratio: 1.4  # Volume surge required
```

### Training Parameters

```yaml
training:
  model_type: "hybrid"      # hybrid, cnn, or lightgbm
  embargo_weeks: 8          # Gap between train/test splits

  cnn:
    lookback_days: 200      # Days of price history per sample
    epochs: 100
    batch_size: 64
    early_stopping_patience: 10
```

### Scanner Settings

```yaml
scanner:
  min_confidence: 0.6       # Minimum score to show
  max_results: 50           # Limit output
```

### Retraining Triggers

```yaml
retraining:
  trigger_new_outcomes: 100     # Retrain after N new outcomes
  drift_threshold: 0.10         # Retrain if metrics drop >10%
```

---

## Project Structure

```
ibd-scanner/
├── config.yaml              # All tunable parameters
├── requirements.txt         # Python dependencies
├── PLAN.md                  # Full architecture documentation
├── CLAUDE.md                # Development conventions
│
├── scanner/                 # Main application
│   ├── config.py           # Config loader
│   ├── db.py               # Database schema + connection
│   ├── universe.py         # Stock universe management
│   ├── data_pipeline.py    # yfinance data fetching
│   ├── breakout_detector.py # Find all breakouts
│   ├── labeler.py          # Historical labeling
│   ├── scan.py             # Live scanner
│   ├── train.py            # Training CLI
│   ├── check_outcomes.py   # Outcome tracker
│   ├── drift_report.py     # Model monitoring
│   │
│   ├── patterns/           # Pattern classifiers (for labeling)
│   │   ├── cup_with_handle.py
│   │   ├── double_bottom.py
│   │   └── flat_base.py
│   │
│   ├── features/           # Feature extractors
│   │   ├── pattern_features.py
│   │   ├── volume_features.py
│   │   ├── rs_features.py
│   │   ├── fundamental_features.py
│   │   └── market_features.py
│   │
│   └── models/             # ML models
│       ├── data_prep.py    # CNN tensor preparation
│       ├── hybrid_model.py # PyTorch CNN + tabular
│       └── training_pipeline.py  # Walk-forward training
│
├── scripts/                 # Standalone utilities
│   ├── backfill.py         # Initial data pull
│   ├── daily_update.py     # Daily updates
│   └── refresh_universe.py # Universe refresh
│
├── labeler/                 # Streamlit labeling UI
│   └── app.py
│
├── tests/                   # Unit tests
│   ├── test_patterns.py
│   ├── test_features.py
│   └── test_pipeline.py
│
├── data/                    # Database storage
│   └── ibd_scanner.db
│
└── models/                  # Saved models
    └── breakout_predictor.pt
```

---

## Troubleshooting

### "No labeled data available"

Run the labeler first:
```bash
python -m scanner.labeler
```

### "No symbols in universe"

Run backfill to populate the database:
```bash
python scripts/backfill.py
```

### "Model not found"

Train the model first:
```bash
python -m scanner.train --model hybrid
```

### yfinance rate limiting

If you see many failed symbols, increase sleep time in `config.yaml`:
```yaml
data:
  yfinance_sleep_seconds: 2  # Increase from 1
```

### Out of memory during training

Reduce batch size in `config.yaml`:
```yaml
training:
  cnn:
    batch_size: 32  # Reduce from 64
```

---

## License

See LICENSE file for details.
