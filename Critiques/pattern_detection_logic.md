# IBD Pattern Detection Logic - Complete Documentation

**Purpose:** This document describes all pattern detection criteria for review. It distinguishes between configurable parameters (in `config.yaml`) and hardcoded logic (in Python files).

---

## Overview

The scanner detects 3 active pattern types:
1. **Cup with Handle** - U-shaped base with optional handle pullback
2. **Double Bottom** - W-shaped base with two lows at similar levels
3. **Flat Base** - Sideways consolidation with tight range

Two patterns are defined but not implemented:
- High Tight Flag (disabled)
- Ascending Base (disabled)

---

## 1. Cup with Handle

**File:** `scanner/patterns/cup_with_handle.py`

### Configurable Parameters (config.yaml)

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Min Depth | `patterns.cup_with_handle.min_depth_pct` | 8% | Minimum decline from left lip to trough |
| Max Depth | `patterns.cup_with_handle.max_depth_pct` | 40% | Maximum decline |
| Min Duration | `patterns.cup_with_handle.min_duration_weeks` | 7 weeks | Minimum pattern length |
| Max Duration | `patterns.cup_with_handle.max_duration_weeks` | 65 weeks | Maximum pattern length |
| Handle Max Depth | `patterns.cup_with_handle.handle_max_depth_pct` | 15% | Max handle pullback |
| Handle Min Weeks | `patterns.cup_with_handle.handle_min_weeks` | 1 week | Min handle duration |
| Handle Max Weeks | `patterns.cup_with_handle.handle_max_weeks` | 4 weeks | Max handle duration |
| Roundness Min Weeks | `patterns.cup_with_handle.roundness_min_weeks_in_bottom` | 3 weeks | Time near bottom |

### Hardcoded Logic (NOT in config)

| Criterion | Value | Location | Description |
|-----------|-------|----------|-------------|
| Peak detection order | 7 | Line 56 | Points on each side for local maxima detection |
| Right lip recovery | 80% | Line 120 | Right lip must reach 80% of left lip price |
| Handle min days | 3 days | Line 179 | Minimum handle duration (overrides config) |
| Handle max depth | 20% | Line 198 | **Conflicts with config 15%** - allows up to 20% |
| Roundness threshold | 5% | Line 240 | Min % of days near trough |
| Symmetry threshold | 0.15 | Line 240 | Min left/right duration ratio |

### Detection Algorithm

```
1. Find all local peaks (order=7) as potential left lips
2. For each left lip:
   a. Find lowest point in [left_lip : left_lip + max_duration_days]
   b. Check depth is between 8-40%
   c. Find right lip (highest point after trough)
   d. Check right lip >= 80% of left lip price
   e. Optionally find handle (pullback after right lip)
   f. Check total duration is 7-65 weeks
   g. Check roundness (not V-shaped):
      - At least 5% of days near trough
      - Left/right symmetry ratio >= 0.15
3. Pivot = high of handle (or right lip if no handle)
```

### Issues Identified

1. **Handle depth conflict:** Config says 15%, code allows 20% (line 198)
2. **Recovery threshold hardcoded:** 80% recovery not configurable
3. **Roundness criteria hardcoded:** 5% near-trough, 0.15 symmetry not configurable

---

## 2. Double Bottom

**File:** `scanner/patterns/double_bottom.py`

### Configurable Parameters (config.yaml)

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Min Depth | `patterns.double_bottom.min_depth_pct` | 8% | Minimum decline from peak to first low |
| Max Depth | `patterns.double_bottom.max_depth_pct` | 40% | Maximum decline |
| Min Duration | `patterns.double_bottom.min_duration_weeks` | 5 weeks | Minimum pattern length |
| Second Low Tolerance | `patterns.double_bottom.second_low_tolerance_pct` | 15% | How different second low can be from first |

### Hardcoded Logic (NOT in config)

| Criterion | Value | Location | Description |
|-----------|-------|----------|-------------|
| Trough detection order | 5 | Line 51 | Points on each side for local minima detection |
| Prior peak lookback | 70 days | Line 81 | Max days to look back for left lip |
| Min prior region | 5 days | Lines 78, 82 | Minimum data before first low |
| Min days to mid-peak | 3 days | Line 96 | Minimum time from first low to mid-peak |
| Max days to mid-peak | 70 days | Line 97 | Maximum time from first low to mid-peak |
| **Mid-peak recovery** | **10%** | **Line 113** | **Rally must recover 10% of the decline** |
| Max days to second low | 70 days | Line 117 | Maximum time from mid-peak to second low |
| Breakout search window | 30 days | Line 146 | Days to look for price breaking pivot |

### Detection Algorithm

```
1. Find all local troughs (order=5) as potential first lows
2. For each first low:
   a. Look back up to 70 days for prior peak (left lip)
   b. Check depth is between 8-40%
   c. Find mid-peak (highest point in 3-70 days after first low)
   d. Check mid-peak rally >= 10% of the decline (HARDCODED)
   e. Find second low (lowest point in 3-70 days after mid-peak)
   f. Check second low is within 15% of first low (CONFIG)
   g. Check total duration >= 5 weeks
3. Pivot = high of mid-peak
4. Find breakout point (first close above pivot within 30 days of second low)
```

### Issues Identified

1. **Critical: Mid-peak recovery hardcoded at 10%** - This is the "rally between lows" requirement but not configurable
2. **15% second low tolerance is too loose** - Allows "double bottoms" where second low is 15% lower than first (not a true W-shape)
3. **Search windows hardcoded** - 70 days for mid-peak and second low not configurable
4. **No check for mid-peak height** - A barely perceptible rally qualifies if it's 10% of the decline

### Example of Bad Classification

VFS was classified as double_bottom with:
- First low: ~$3.12 (around Mar 6)
- Mid-peak: ~$3.16 (tiny rally)
- Second low: ~$2.78 (11% lower than first)

This passed because:
- 11% < 15% tolerance
- Tiny rally was > 10% of small decline

But visually it's a descending base, not a W-shape.

---

## 3. Flat Base

**File:** `scanner/patterns/flat_base.py`

### Configurable Parameters (config.yaml)

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Max Depth | `patterns.flat_base.max_depth_pct` | 15% | Maximum high-to-low range |
| Min Duration | `patterns.flat_base.min_duration_weeks` | 5 weeks | Minimum consolidation time |

### Hardcoded Logic (NOT in config)

| Criterion | Value | Location | Description |
|-----------|-------|----------|-------------|
| Max window multiplier | 3x | Line 49 | Looks for bases up to 3x min duration (15 weeks) |
| Step size | 5 days | Line 51 | Window slides by 1 week at a time |
| Tightness CV threshold | 10% | Line 154 | Coefficient of variation for "loose" |
| Min data points | min_days + 10 | Line 41 | Minimum bars needed |

### Detection Algorithm

```
1. Slide a window across price data (step = 5 days)
2. For each starting position:
   a. Try window sizes from min_duration to 3x min_duration
   b. Calculate range_pct = (high - low) / low * 100
   c. If range_pct <= 15% (max_depth):
      - Calculate tightness score (CV of weekly closes)
      - Keep expanding window while still valid
   d. Keep the largest valid flat base
3. Deduplicate overlapping patterns (keep higher confidence)
4. Pivot = high of the base
```

### Tightness Score Calculation

```python
# Weekly closes sampled every 5 days
cv = std(weekly_closes) / mean(weekly_closes)
tightness = max(0, min(1, 1 - (cv / 0.10)))
# CV < 2% -> tightness ~ 0.8
# CV = 10% -> tightness = 0
# CV > 10% -> tightness = 0
```

### Issues Identified

1. **No minimum depth** - Could detect flat bases during downtrends
2. **No prior uptrend requirement** - Flat bases should follow an advance
3. **Tightness score not used as filter** - Only affects confidence, not detection

---

## 4. Base Detector (Shared Logic)

**File:** `scanner/patterns/base_detector.py`

### Shared Methods

| Method | Description |
|--------|-------------|
| `find_peaks(prices, order)` | Uses `scipy.signal.argrelextrema` with `np.greater_equal` |
| `find_troughs(prices, order)` | Uses `scipy.signal.argrelextrema` with `np.less_equal` |
| `smooth(prices, window)` | Simple moving average |
| `compute_depth_pct(peak, trough)` | `(peak - trough) / peak * 100` |
| `trading_days_to_weeks(days)` | `days / 5.0` |

### Global Config

| Parameter | Config Key | Default |
|-----------|------------|---------|
| Sensitivity | `patterns.sensitivity` | 0.5 |

Sensitivity is used by `adjust_threshold()` but **not currently used** by any detector.

---

## 5. Disabled Patterns

### High Tight Flag
- **Status:** Not implemented (placeholder file)
- **Planned criteria:** 100%+ run-up followed by 10-25% pullback

### Ascending Base
- **Status:** Not implemented (placeholder file)
- **Planned criteria:** Three pullbacks with rising lows during market correction

---

## Summary of Issues

### Hardcoded Values That Should Be Configurable

| Pattern | Parameter | Current Value | Impact |
|---------|-----------|---------------|--------|
| Cup | Right lip recovery | 80% | Could miss valid cups with 75% recovery |
| Cup | Handle max depth | 20% (code) vs 15% (config) | Inconsistency |
| Cup | Roundness thresholds | 5%, 0.15 | May reject valid rounded cups |
| Double Bottom | Mid-peak recovery | 10% | Key criterion not tunable |
| Double Bottom | Search windows | 70 days | May miss longer patterns |
| Flat Base | Max window multiplier | 3x | Limits detection to 15 weeks |

### Logic Issues

1. **Double Bottom 15% tolerance is too loose** - Allows second low to be much lower than first, which isn't a true W-shape
2. **No minimum mid-peak height for Double Bottom** - A tiny blip qualifies as the "rally"
3. **Flat Base has no prior uptrend check** - Could detect consolidation in downtrends
4. **Sensitivity config is unused** - The `adjust_threshold()` method exists but no detector calls it

---

## Recommendations for Review

1. Should second_low_tolerance be tightened to 5-8%?
2. Should mid-peak recovery be increased from 10% to 30-50%?
3. Should there be a minimum mid-peak height (e.g., must reach 50% of left lip)?
4. Should flat bases require a prior uptrend?
5. Should the roundness/symmetry criteria be configurable?
