# IBD Scanner Experimentation Log

## Status: RUNNING

## Current Best
- Experiment: #1 (Logistic Regression Baseline)
- AUC: 0.570
- Precision: 0.270
- Recall: 0.524

## Stopping Conditions
- [ ] AUC > 0.65 AND Precision > 0.35 → STOP (target reached)
- [ ] 5 experiments without improvement → STOP (diminishing returns)
- [ ] User writes STOP → STOP

## Experiment Queue
1. [DONE] Logistic regression baseline diagnostic
2. [DONE] Symmetric labels (+10%/-10%) - AUC 0.53, F1 improved 5.8x
3. [DONE] Add tightness feature (pre_breakout_tightness, pre_breakout_range_pct)
4. [DONE] Add RS acceleration feature (rs_acceleration)
5. [READY] Multi-label experiments - All 3 strategies computed in one pass
6. [READY] Focal loss - Implemented in training pipeline
7. [PENDING] Multi-seed ensemble (5 seeds)

## New Multi-Label Framework (Implemented 2026-04-05)

Database now stores 3 outcome variants per pattern:
- `outcome_asym_20_7`: Original IBD +20%/-7% (12-week window)
- `outcome_asym_15_10`: Less extreme +15%/-10% (12-week window)
- `outcome_sym_10`: Symmetric +10%/-10% (12-week window)

**To run experiments:**
```bash
# Step 1: Re-run labeler to compute all 3 label variants
python -m scanner.labeler --force

# Step 2: Run comparison experiments
python scripts/run_label_experiments.py

# Or run individual experiments:
python -m scanner.train --model lightgbm --label-strategy asym_20_7 --focal-loss
python -m scanner.train --model lightgbm --label-strategy asym_15_10
python -m scanner.train --model lightgbm --label-strategy sym_10
```

## Experiment History

| # | Name | AUC | Prec | Recall | F1 | Notes |
|---|------|-----|------|--------|-----|-------|
| 1 | LogReg Baseline (asymmetric labels) | 0.570 | 0.270 | 0.524 | 0.354 | Original +20%/-7% labels |
| 2 | Symmetric labels +/-10% | 0.531 | 0.510 | 0.494 | 0.496 | Balanced classes (51% success) |
| 2 | LightGBM (symmetric labels) | 0.525 | 0.504 | 0.641 | 0.543 | Balanced, F1 improved 5.8x |

## Key Findings

### Experiment #1: Logistic Regression Diagnostic (2026-04-05)

**Purpose:** Establish if any signal exists before investing in complex models.

**Results (asymmetric labels):**
- AUC 0.57 beats random (0.50) by meaningful margin
- Wide split variation: Split 7 = 0.64 AUC, Split 2 = 0.52 AUC

### Experiment #2: Symmetric Labels +/-10% (2026-04-05)

**Purpose:** Balance classes to remove artificial signal from class imbalance.

**Results:**
- Success rate: 23% → 51% (balanced!)
- LogReg AUC: 0.57 → 0.53 (signal was partly from imbalance)
- Precision: 0.27 → 0.51 (more meaningful predictions)
- LightGBM F1: 0.09 → 0.54 (5.8x improvement!)

**Key Insight:** The "signal" in asymmetric labels was partly the model exploiting the 77% failure rate. True predictive power is weak (AUC ~0.53). Need better features.

**Feature Coefficients (symmetric):**
- `market_cap_log`: +0.139 (larger caps better)
- `base_depth_pct`: +0.068 (deeper bases → more success)
- `breakout_volume_ratio`: -0.048 (still negative!)
- `rs_rank_percentile`: +0.044 (higher RS better)
- `sp500_trend_4wk`: -0.030 (still negative!)

---

## User Commands
(Write commands here - supervisor will read them)

---

## Next Steps

**COMPLETED:**
1. Symmetric labels implemented and applied (+10%/-10%)
2. New features added to code:
   - `pre_breakout_tightness` - ratio of final 3-week range to earlier range
   - `pre_breakout_range_pct` - final range as % of price (ATR-like)
   - `rs_acceleration` - 2nd derivative of RS line (recent slope minus prior slope)

**TO DO:**
3. **Re-run labeler** to populate new features:
   ```bash
   python -m scanner.labeler --force
   ```
   This will take ~15 minutes but will compute the new features.

4. **User can continue labeling gold standard** (Streamlit app still works)

5. **After labeler finishes:** Run diagnostic to measure impact of new features

6. **Future experiments:**
   - Multi-seed ensemble
   - Focal loss
   - Earnings proximity (requires external data)
   - Sector momentum (requires sector classification)

---

## Data Summary

| Metric | Value |
|--------|-------|
| Total patterns | 31,311 |
| Labeled patterns | 29,797 |
| Success rate | 23.2% |
| Baseline AUC | 0.570 (LogReg) |
| Target AUC | > 0.65 |
