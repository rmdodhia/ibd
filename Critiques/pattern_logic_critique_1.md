Overall, the logic is directionally right, though it is too loose in a few places and missing several IBD-defining constraints. The biggest issue is not that the code is detecting the wrong families of patterns. It is that the current criteria will admit too many borderline or visually flawed cases, especially for double bottoms and cups with weak right sides. That kind of label contamination can easily wash out model signal. Your own documentation already flags several of the weak spots, and I agree with most of those flags.

My short verdict by pattern:

Cup with handle: concept is mostly sound, though the implementation is too permissive and mixes cup-with-handle with cup-without-handle.
Double bottom: this is the weakest of the three. The current rules do not enforce a convincing W.
Flat base: the sideways-range logic is reasonable, though it is missing one of the most important IBD conditions, namely a prior uptrend.

For IBD specifically, a proper cup with handle is typically at least seven weeks long, the cup depth is commonly in the roughly 12% to 33% range, and the handle should form in the upper half of the base. IBD also treats the handle as a meaningful structural feature, not just an optional decoration.

Against that standard, your cup logic needs tightening. The file says the detector allows cup depth from 8% to 40%, right-lip recovery of only 80% of the left-lip price, a handle depth conflict between config 15% and code 20%, and “optional handle” behavior where the pivot can simply be the right lip. The 8% floor is below normal IBD guidance, and 40% is broad enough to admit many deep, damaged structures. More importantly, an 80% right-lip recovery is too lax, not too strict. If the left lip is 100 and the right lip only gets back to 80, the pattern is still 20% below the old high, which is generally too low for a proper handle zone. IBD’s guidance is closer to “handle in the upper half” and often “peak of the handle within about 15% of the old high.”

So for cup-with-handle, I would change the logic this way:

Require a prior uptrend before the base starts.
Tighten cup depth to something like 12% to 33% for standard conditions.
Replace “right lip >= 80% of left lip” with an upper-half test. A simple version is
right_lip >= trough + 0.5 * (left_lip - trough)
though for IBD purity I would make it stricter, closer to within 10% to 15% of the old high.
If the classifier says cup with handle, then require an actual handle. Put cup-without-handle in a separate class.
Tighten handle depth toward 10% to 12% in normal bull conditions, while allowing a looser regime-specific override only if you explicitly want post-bear-market flexibility.

The roundness logic is serviceable as a first pass, though it is a blunt proxy. Your current “5% of days near trough” plus symmetry ratio 0.15 does at least try to reject V-shapes. I would keep the idea, though make it configurable and stronger. A better detector would also use either smoothed prices, curvature, or a requirement that both the left and right side each take a minimum number of bars. Right now the symmetry threshold looks too weak to carry much weight.

On double bottom, the decision logic is the least sound of the three. IBD’s definition is more specific than “two lows at similar levels.” The second leg should undercut the first low, creating the classic W and shaking out weak holders. If it does not undercut, IBD itself says to treat it cautiously and not really call it a proper double bottom.

Your file says the detector allows the second low to be “within 15%” of the first low, and only requires the midpoint rally to recover 10% of the decline. That is far too weak. It allows a descending base with a tiny bump in the middle to count as a W, which is exactly what your VFS example shows.

This is where I would be much stricter:

Require the second low to undercut the first low by a small amount, maybe 0.5% to 3% as a starting range.
Add an upper cap so the undercut is not too deep, maybe no more than 5% to 8% below the first low.
Require the midpoint peak to be a real peak, not just a tiny bounce. A practical rule is that the midpoint should be at least 5% to 10% above both lows, and preferably visually prominent relative to recent noise.
Keep the pivot at the midpoint peak, which is consistent with IBD.

So your documentation’s instinct is right here: the 15% tolerance is too loose, and the missing minimum midpoint prominence is a serious flaw.

On flat base, the range logic is mostly aligned with the idea of a tight sideways consolidation. IBD describes a flat base as a base after a prior uptrend, usually at least five weeks long, and no more than about 15% deep. It often forms after a prior breakout or after a decent prior run.

Your file shows you already enforce max depth 15% and min duration 5 weeks, which is good. The missing prior-uptrend condition is the big omission. Without it, the detector can find sideways pauses in weak or falling stocks and call them flat bases, which is not the IBD idea. Your note about tightness only affecting confidence, not eligibility, is also correct. “Wide and loose” action should usually be filtered out, not merely scored lower.

So for flat base I would require:

Prior uptrend, ideally something like 20% to 30% before the base.
Depth <= 15%.
Duration >= 5 weeks.
A tightness threshold as an actual gate.
Optional support checks such as holding above a rising 10-week line or at least not violating it badly.

Across all three patterns, one major conceptual gap is the prior uptrend requirement. IBD explicitly emphasizes that bases form after meaningful advances, often around 30% or more. Without that, you are classifying shapes in isolation, which is not really IBD pattern recognition.

I also think the code has a design issue beyond the trading logic: too many important criteria are hardcoded, and one config parameter conflicts with code. Your file shows that for cups, the config says 15% handle depth while code allows 20%, and for multiple patterns the key thresholds are hardwired rather than configurable. It also notes that the shared sensitivity parameter is unused. That is not just untidy. It makes calibration harder and can produce a false sense that config.yaml is the source of truth when it is not.

So, directly on your question of whether the decision logic is captured by the decision criteria:

Cup with handle: partially. The criteria capture the broad idea, though they do not capture the upper-half handle requirement well enough, and they blur cup-with-handle with cup-without-handle.
Double bottom: not well enough. The current criteria do not faithfully capture the W-shape because they do not require a meaningful central peak or a proper undercut structure.
Flat base: mostly, though the omission of prior uptrend means the criteria are missing a defining part of the setup.

If I had to rank fixes by importance:

Add prior uptrend checks to all three patterns.
Split cup-with-handle from cup-without-handle.
Tighten double-bottom so the second low must undercut the first and the midpoint must be visually meaningful.
Replace the cup’s 80% right-lip recovery with a stricter upper-half or near-old-high condition.
Make “wide and loose” a hard fail for flat bases.
Move all important thresholds into config and eliminate code/config conflicts.

The single biggest flaw is the double-bottom logic. The second biggest is the cup logic allowing a right side that is too low to qualify as a proper handle setup. Those two alone could generate a lot of noisy positives and help explain why the downstream model is struggling.