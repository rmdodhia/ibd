# Review of IBD Pattern Detection Logic

Based on the uploaded detector summary, the logic is directionally right, but it is too loose in several places and misses some core IBD constraints. The main issue is not that the code is targeting the wrong pattern families. The bigger issue is that the current criteria admit too many borderline or visually flawed cases, especially for double bottoms and cups with weak right sides. That kind of contamination can easily weaken downstream model signal. See the uploaded detector summary for the documented thresholds and hardcoded rules. fileciteturn1file0

## Overall verdict

- **Cup with handle:** mostly sound in concept, but too permissive and it blurs cup-with-handle with cup-without-handle.
- **Double bottom:** the weakest of the three. The current rules do not enforce a convincing W-shape.
- **Flat base:** the sideways consolidation logic is reasonable, but it is missing one of the defining IBD conditions, namely a prior uptrend.

## 1. Cup with handle

IBD-style cup-with-handle patterns are typically at least seven weeks long, the cup depth is often around 12% to 33%, and the handle should form in the upper half of the base. The handle is a meaningful structural part of the setup, not just an optional add-on.

Your detector, as documented, allows:

- cup depth from 8% to 40%
- right-lip recovery of only 80% of the left-lip price
- a conflict between config handle depth 15% and code handle depth 20%
- optional handle behavior where the pivot can be the right lip

These settings are too loose for a clean IBD interpretation. The biggest issue is the right-lip recovery rule. If the left lip is 100 and the right lip only gets back to 80, the pattern is still 20% below the old high. That is generally too low for a proper handle zone.

### Assessment

The broad concept is correct, but the current criteria do **not** capture a high-quality cup-with-handle tightly enough.

### Improvements

1. Require a **prior uptrend** before the base starts.
2. Tighten cup depth to roughly **12% to 33%** under normal conditions.
3. Replace `right_lip >= 80% of left_lip` with a stronger rule. Two reasonable options:
   - right lip must be in the **upper half** of the base
   - right lip must be **within 10% to 15% of the old high**
4. If the class is **cup with handle**, require an actual handle. Put cup-without-handle in a separate class.
5. Tighten handle depth toward **10% to 12%** in normal market conditions.
6. Keep the roundness logic, but make it configurable and stronger.

## 2. Double bottom

This is the least sound of the three.

A proper IBD double bottom is more specific than two lows at roughly similar levels. The second low should usually **undercut the first low slightly**, creating the classic W-shape and shaking out weak holders. The middle peak should also be a real rally, not just a small bounce.

Your documentation says the detector allows:

- second low within 15% of the first low
- midpoint rally only 10% of the prior decline
- no explicit requirement that the midpoint be visually prominent

That is far too permissive. It allows a descending structure with a tiny mid-bounce to qualify as a double bottom.

### Assessment

The decision logic does **not** faithfully capture an IBD double bottom.

### Improvements

1. Require the second low to **undercut** the first low by a small amount, perhaps **0.5% to 3%** as a starting range.
2. Cap the undercut so it is not too deep, perhaps no more than **5% to 8%** below the first low.
3. Require the midpoint peak to be a **real peak**, for example at least **5% to 10% above both lows**.
4. Keep the pivot at the midpoint high.
5. Tighten or redesign the current `second_low_tolerance_pct` logic. A simple “within 15%” rule is not enough.

## 3. Flat base

The flat-base detector is the closest to the intended IBD concept, but it still misses one important condition.

A flat base is a tight sideways consolidation after a prior uptrend, often lasting at least five weeks and staying within a relatively narrow range, often around 15% or less in depth.

Your detector already enforces:

- max depth of 15%
- minimum duration of 5 weeks

That part is reasonable. The main problem is the lack of a **prior uptrend requirement**. Without it, the code can detect sideways pauses in weak or declining stocks and call them flat bases.

### Assessment

The price-range logic is reasonable, but the decision criteria are incomplete.

### Improvements

1. Require a **prior uptrend**, for example a 20% to 30% move before the base.
2. Keep the current depth and duration tests.
3. Make tightness a **hard filter**, not just a confidence score.
4. Optionally add checks related to support at key moving averages.

## Cross-pattern issues

### Prior uptrend is missing

This is one of the biggest conceptual omissions. IBD bases are not just shapes. They are continuation or consolidation structures that typically occur after meaningful advances.

### Hardcoded logic should move into config

Your uploaded summary identifies several key thresholds that are hardcoded rather than configurable, including:

- cup right-lip recovery
- cup roundness thresholds
- double-bottom mid-peak recovery
- several search windows

There is also a config/code conflict for handle depth in the cup pattern, and the global sensitivity setting is currently unused. That makes the system harder to calibrate and review. fileciteturn1file0

### Detector purity matters more than model complexity

If these pattern rules admit too many poor examples, the downstream ML model will be learning from noisy positives and noisy negatives. Tightening the detector logic is likely to help more than changing the model architecture.

## Direct answer to your question

### Is the decision logic sound?

Partly.

- **Cup with handle:** broadly sound, but too loose.
- **Double bottom:** not sound enough for an IBD-style W.
- **Flat base:** mostly sound, but incomplete because prior uptrend is missing.

### Is the logic captured by the current decision criteria?

Only partially.

- **Cup with handle:** the current criteria capture the broad idea, but they do not enforce the upper-half handle zone tightly enough, and they blur cups with and without handles.
- **Double bottom:** the current criteria do not capture the defining W-shape well enough.
- **Flat base:** the current criteria capture the sideways range idea, but omit a defining context condition.

## Priority fixes

1. Add **prior uptrend** checks to all three patterns.
2. Split **cup-with-handle** from **cup-without-handle**.
3. Tighten **double-bottom** so the second low must undercut the first and the midpoint must be meaningful.
4. Replace the cup’s **80% right-lip recovery** with a stricter near-old-high or upper-half rule.
5. Make “wide and loose” a **hard fail** for flat bases.
6. Move all important thresholds into config and remove code/config conflicts.

## Bottom line

The single biggest flaw is the **double-bottom logic**. The second biggest is the cup logic allowing a right side that is too low to qualify as a proper handle setup. Those two issues alone could create many noisy detections and degrade model training quality.
