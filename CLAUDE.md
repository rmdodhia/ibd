# CLAUDE.md — Context for Claude Code CLI

## Project: IBD Breakout Scanner

A Python-based stock screening system that detects IBD/CAN SLIM chart patterns, predicts breakout success using ML, and improves over time.

## Key Files

- `PLAN.md` — Full project plan with architecture, phases, schemas, and checklists
- `config.yaml` — All tunable parameters (thresholds, universe, ML settings)
- `requirements.txt` — Python dependencies

## Architecture

- **Language:** Python 3.11+
- **Database:** SQLite (`data/ibd_scanner.db`)
- **Data source:** yfinance (free)
- **ML:** LightGBM (Phase 3a), PyTorch 1D CNN (Phase 3b), Hybrid (Phase 3c)
- **Labeling UI:** Streamlit
- **Config:** YAML-driven, all thresholds in `config.yaml`

## Code Conventions

- Use type hints on all functions
- Docstrings on all public functions (Google style)
- Use `scanner.db.get_connection()` for all database access
- Load config via `scanner.config.load_config()` — never hardcode thresholds
- Pattern detectors inherit from a common base class in `patterns/base_detector.py`
- Feature extractors are pure functions: take price DataFrame + metadata, return dict of features
- All dates stored as ISO 8601 strings in SQLite
- Use pandas DataFrames for price data manipulation
- Use numpy for numerical computations
- Logging via Python `logging` module, not print statements

## Commands

```bash
# Setup
pip install -r requirements.txt

# Data pipeline
python scripts/backfill.py                    # Initial 10yr data pull
python scripts/daily_update.py                # Daily incremental update
python scripts/refresh_universe.py            # Update stock universe

# Labeling
python -m scanner.labeler                     # Run historical labeler
streamlit run labeler/app.py                  # Launch labeling UI

# Training
python -m scanner.train --model lightgbm      # Train LightGBM
python -m scanner.train --model cnn           # Train CNN
python -m scanner.train --model hybrid        # Train hybrid
python -m scanner.train --compare-to v1.0     # Compare to previous version

# Scanning
python -m scanner.scan                        # Full universe scan
python -m scanner.scan --symbol AAPL          # Single stock
python -m scanner.scan --min-confidence 0.7   # Filter by confidence

# Monitoring
python -m scanner.check_outcomes              # Update prediction outcomes
python -m scanner.drift_report                # Model performance report

# Testing
pytest tests/
```

## Database Schema

See PLAN.md section 1.3 and 2.4 for full schema. Key tables:
- `stocks` — symbol metadata
- `daily_prices` — OHLCV data (primary key: symbol + date)
- `fundamentals` — quarterly EPS/revenue
- `detected_patterns` — auto + human labeled patterns
- `pattern_features` — extracted features per pattern
- `predictions` — live predictions with outcomes
- `model_runs` — training run metadata

## Current Phase

Starting Phase 1 — Data Pipeline. See PLAN.md task checklist for current status.

## Important Notes

- Always read thresholds from config.yaml, never hardcode
- yfinance has rate limits — use batching + sleep (see config.yaml)
- Walk-forward validation only — never use future data in training splits
- Pattern detectors should be independent and composable
- The training pipeline must be re-runnable (same command for initial train and retrain)
- All model evaluation uses precision, recall, F1, and profit factor
