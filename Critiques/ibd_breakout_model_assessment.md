# IBD breakout model assessment

My read is that this is **probably solvable enough to be useful**, though likely **not** as a high-accuracy, standalone “discover hidden breakout laws from charts” problem. The strongest version is: use IBD rules to generate candidate setups, then train a model to **rank or gate** those candidates. That fits your current pipeline much better than asking one model to infer everything from noisy pattern detections and a brittle binary label. Your current setup has about 31k labeled patterns, a 23% success rate, and AUC around 0.53 to 0.55, so there is little usable signal in the present formulation. fileciteturn0file0

The first thing I would change is the **problem decomposition**.

You currently have at least two distinct tasks mixed together. One is “does this chart form a proper IBD base?” The other is “conditional on a breakout trigger, is this trade likely to work?” Those are different targets with different labels and different useful features. Your detector is rule-based already, so I would keep that as the first-stage filter and train the ML model as a **meta-labeler** on top of it. That is very close to the financial ML idea of using a white-box primary rule and a secondary model to decide which signals to take.

On your question of whether the problem is fundamentally solvable, I would say **yes, though only modestly**. A 23% base success rate does not imply impossibility. It means the signal-to-noise ratio is low. Finance datasets are widely recognized as low-signal, prone to survivorship bias, and easy to overfit. So an AUC target above 0.75 is, in my view, too optimistic for this setup. A model that materially improves ranking quality, precision in the top decile, or net expectancy can still be valuable even if headline AUC stays moderate.

On architecture, I would **de-prioritize it**.

The fact that your CNN and LightGBM are both near random is a strong hint that the bottleneck is **labels, event definition, leakage control, or detector quality**, not missing model capacity. InceptionTime is a credible next architecture for time-series classification, and it has a stronger track record than generic small CNNs. A ResNet1D is also reasonable. A Transformer would be later on my list. With an effective sample size that is smaller than 31k once you account for dependence and overlap, I would expect a Transformer to be easier to overfit and harder to justify.

I also think you should **tighten the detector before training anything deep**. Your own notes already point in that direction, and I agree. IBD’s own published guidance puts proper cup depth roughly in the 12% to low-30% range, requires at least seven weeks, and expects the handle in the upper part of the base with handle depth roughly under 8% to 12%. Your current thresholds are looser than that, so your first-stage candidate set is likely contaminated with many weak or invalid patterns. If the detector is loose, the model spends its capacity sorting noise rather than learning follow-through.

The biggest missing item in the plan is **dataset realism**.

From the description, you are pulling about 10 years of history for a filtered current universe from Yahoo Finance. That raises a likely survivorship-bias concern. Dead names, acquired names, and historically illiquid names may be underrepresented or absent. In breakout trading, that matters a lot because the losers often disappear from the modern universe. I would also worry about whether fundamentals are truly point-in-time and whether repeated overlapping events for the same ticker are inflating your effective sample count. I’m inferring that risk from your setup, not stating that it has already happened. fileciteturn0file0

Related to that, your current validation is decent, though I would still upgrade it. You already use walk-forward splits with an 8-week embargo. That is a good start. In financial ML, overlapping event windows are a classic source of leakage, and purged validation with event end times is preferred over plain walk-forward when labels overlap.

On the success definition, here is my view.

Your current label is **trader-legible** because it echoes IBD’s sell logic. That said, it is still **not the best ML label**. It mixes a discretionary trading doctrine with a hard supervised-learning target. I would test a **volatility-scaled triple-barrier label** instead of fixed +20% and -7% thresholds. In practice, that means the upper and lower barriers depend on ATR or realized volatility, and the horizon is specified in trading days. That aligns the label with the stock’s own volatility regime and usually makes cross-sectional learning more coherent. I would run both versions in parallel: the IBD-consistent label for interpretability, and a triple-barrier label for model quality.

I would also simplify first. Very much so.

Before touching ResNets, I would build four baselines:

1. A **rule-only** baseline using your current pattern score and a few hard filters.
2. A **logistic regression** with maybe 8 to 12 clean features.
3. A **gradient-boosted tree** with only pre-breakout and breakout-day features you can actually know at decision time.
4. A **meta-labeling model** on top of high-confidence detector outputs only.

If those do not beat the current level, a deeper encoder will probably not rescue the project.

A few parts of your plan I would re-rank:

- **Move up:** detector precision audit, point-in-time data audit, leakage audit, simpler baselines, label redesign.
- **Move down:** seasonality as a “quick win.” Month effects can be real in-sample and useless out of sample.
- **Move down:** focal loss, label smoothing, dropout tweaking. These are fine cleanup steps, though I would not expect them to create signal.
- **Move up:** metrics that reflect trading use, such as PR-AUC, precision at top-k, top-decile return, calibration, and performance by market regime.

If I were running this project, my first sprint would look like this:

1. Build a **gold set** of a few hundred manually verified patterns with strict IBD criteria.
2. Score the detector against that set.
3. Rebuild labels with **event-based triple barriers** and point-in-time features.
4. Switch evaluation to **purged event-based validation**.
5. Run simple baselines.
6. Only then try InceptionTime or a small ResNet1D.

So, direct answers to your numbered questions:

1. **Solvable?** Yes, in a limited ranking sense. No evidence yet that it supports a strong standalone predictor.
2. **Will ResNet/InceptionTime help?** Possibly, though only after fixing labels and event definition. InceptionTime is the first one I’d try.
3. **Is the success definition appropriate?** Reasonable for IBD-style trading rules, weak as the only ML target.
4. **What’s missing?** Point-in-time data, survivorship bias control, event-overlap leakage control, meta-labeling, ranking metrics, and decision-time discipline.
5. **Simplify first?** Yes.
6. **Data quality or model quality first?** Data and labels first, by a wide margin.
7. **Contrastive pretraining overkill?** For now, yes.

The single highest-value change is this: **treat the model as a gatekeeper for rule-based breakouts, not as the primary discoverer of chart patterns.**
