# IBD Breakout Scanner — Critique & Recommendations

I've read through your experimentation plan carefully. This is a well-structured project with honest diagnostics. Let me work through your questions and offer some strategic guidance.

---

## Is the problem fundamentally solvable?

Partially — but you need to reset expectations. Breakout prediction from chart patterns alone is a weak-signal problem. An AUC of 0.60–0.65 with precision around 0.35–0.45 would actually be a meaningful result in this domain, because even a modest edge compounds over many trades. The fact that both CNN and LightGBM land at ~0.53–0.55 suggests there's *some* signal but it's being drowned by noise — not that the problem is hopeless.

The key question is whether your current features and labels are capturing the right thing. I think several aren't, and that's where your biggest gains will come from.

---

## Prioritize data quality over model architecture — decisively

Your plan proposes both data fixes and architecture upgrades. **Do the data work first and don't touch the architecture until you've exhausted data improvements.** Here's why:

CNN and LightGBM performing nearly identically is the clearest possible signal that model capacity isn't the bottleneck. LightGBM is a powerful learner — if it can't find signal in your tabular features, adding skip connections to a CNN won't help. ResNet, InceptionTime, and Transformers are solutions to underfitting and limited receptive fields, but your model isn't underfitting — it's learning nothing because the labels and features aren't giving it anything to learn.

Your Day 3 (architecture) should become Day 5+. Your Day 1–2 (data) should expand to fill the first week.

---

## The success definition is your single biggest problem

The current +20%/−7% asymmetry is severely biasing your labels. A stock only needs to dip 7% to be labeled a failure but needs to rally 20% to be a success — that's a nearly 3:1 asymmetry, which directly explains your 23% success rate. You're not measuring breakout quality; you're measuring whether a stock avoids any normal pullback for 8 weeks while also tripling the loss threshold in gains.

Your Option A (+15% in 6 weeks with <5% max drawdown) is better but still has issues. The 5% drawdown constraint is very tight — most successful breakouts pull back to the pivot point, which can easily be a 3–7% move.

**My recommendation:** Use a risk-adjusted metric instead. Define success as achieving a reward/risk ratio ≥ 2:1, where risk = max adverse excursion from entry, and reward = max favorable excursion within the outcome window. This aligns with how traders actually evaluate setups. Alternatively, use a simpler symmetric threshold like +10%/−10% in 8 weeks as a starting point — it'll get your success rate closer to 35–40%, which gives the model much more to work with.

Also consider making this a regression problem (predict the reward/risk ratio directly) rather than binary classification. You lose less information that way.

---

## Start simpler — yes, absolutely

Before anything else, train a logistic regression on your top 5 LightGBM features: `sp500_trend_4wk`, `base_depth_pct`, `market_cap_log`, `rs_rank_percentile`, `breakout_volume_ratio`. If logistic regression with 5 features can't beat AUC 0.55, that tells you definitively that either the labels are wrong or the features don't capture the signal. No amount of architectural complexity will fix that.

This takes 20 minutes and gives you a critical diagnostic. Do it before anything in your current plan.

---

## What's missing from the plan

A few things I'd add:

**Proper feature for "tightness of price action before breakout."** IBD practitioners emphasize that the last 2–3 weeks before breakout should show contracting range and declining volume. You have `up_down_volume_ratio` but not a direct measure of price range contraction in the final handle/base area. This is arguably the single most important IBD signal and it's underrepresented in your features.

**Relative strength vs. the market at the time of breakout, not just the RS rank.** The *acceleration* of RS (is it improving into the breakout?) matters more than the level. You have `rs_line_slope_4wk` but consider adding slope change (second derivative) — is RS accelerating?

**Earnings proximity.** Breakouts that occur within 2–3 weeks before earnings behave very differently from those with no upcoming catalyst. If you can add a binary "earnings within 3 weeks" feature, it could be quite informative.

**Sector momentum.** Not just sector as a categorical — but whether the stock's sector is in the top 20% of 3-month performance. IBD's own methodology emphasizes sector/industry group rank heavily.

---

## On your specific architecture proposals

If you do eventually get to architecture work after fixing data:

- **ResNet1D with SE blocks (Option A):** Reasonable first step, low effort, worth trying. But don't expect more than +0.02–0.03 AUC if data is the real issue.
- **InceptionTime (Option B):** This is actually the most theoretically justified for your problem, since cup-with-handle patterns span different time scales. I'd rank this above ResNet.
- **Transformer (Option C):** Likely overkill for 200-timestep sequences with 31k samples. You don't have enough data for attention to shine.
- **GAF/2D Image (Option D):** Skip this entirely. It adds complexity without clear benefit for 1D financial time series. The literature on GAF for stock prediction is not compelling.
- **Contrastive pretraining:** Not overkill conceptually, but premature. Revisit only if you get AUC > 0.60 with supervised methods and want to squeeze out more.

---

## On focal loss and training tweaks

Focal loss is a good idea and quick to implement — do it. But understand what it does and doesn't solve. Focal loss helps when your model is lazy and only predicts the majority class. If your model is already predicting some positives (recall=0.54 for the CNN), focal loss will have modest impact. It's most useful when recall is near zero.

Label smoothing and increased dropout are fine but incremental. The cosine LR schedule with warmup is good practice but won't rescue a model that lacks signal.

**The ensemble idea (multi-seed averaging) is actually one of the highest-value items on your list.** It's trivial to implement and typically gives +0.02–0.04 AUC for free. Move it earlier in your plan.

---

## Revised implementation order

### Week 1: Diagnostics & Data

1. Logistic regression on top 5 features (diagnostic baseline)
2. Fix success definition (symmetric or risk-adjusted)
3. Re-label all patterns with new definition
4. Add features: tightness before breakout, RS acceleration, earnings proximity, sector momentum, seasonality
5. Retrain LightGBM — if AUC < 0.58, the features still aren't right

### Week 2: Training improvements

6. Focal loss + label smoothing
7. Feature scaling (StandardScaler)
8. Multi-seed ensemble (5 seeds, average predictions)
9. Cosine LR schedule with warmup

### Week 3: Architecture (only if Week 1–2 gets AUC > 0.58)

10. InceptionTime (multi-scale, most justified for your problem)
11. Stacking ensemble (CNN embeddings → LightGBM)

### Week 4+: Advanced (only if AUC > 0.65)

12. Contrastive pretraining
13. Regression target (reward/risk ratio)

---

## Bottom line

Your project is well-built and your diagnostics are honest — that puts you ahead of most ML projects. The core issue is that you're trying to solve a data quality problem with model complexity. Fix the labels, add the right features (especially tightness and RS acceleration), validate signal with a simple model, and only then layer on architectural sophistication. The most likely path to AUC > 0.65 runs through better labels and features, not through Transformers.
