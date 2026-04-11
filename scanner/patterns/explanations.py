"""Pattern confidence explanation generator.

Reconstructs the scoring logic from each detector's _compute_confidence() method
to provide human-readable explanations of why a pattern scored its confidence level.
"""

from typing import Optional


# Scoring criteria for each pattern type
# Format: (factor_name, check_function, bonus, description)
# check_function takes metadata and returns (met: bool, value_str: str)

CUP_WITH_HANDLE_CRITERIA = [
    {
        "factor": "Depth (ideal 15-25%)",
        "key": "depth_pct",
        "tiers": [
            {"range": (15, 25), "bonus": 0.12, "desc": "Ideal depth range"},
            {"range": (12, 35), "bonus": 0.06, "desc": "Acceptable depth range"},
        ],
    },
    {
        "factor": "Duration (ideal 7-30 weeks)",
        "key": "duration_weeks",
        "tiers": [
            {"range": (7, 30), "bonus": 0.08, "desc": "Good duration range"},
        ],
    },
    {
        "factor": "Has Handle",
        "key": "has_handle",
        "boolean": True,
        "bonus": 0.10,
        "desc": "Pattern includes handle formation",
    },
    {
        "factor": "Handle Depth (ideal <10%)",
        "key": "handle_depth_pct",
        "condition": "has_handle",  # Only applies if has_handle is True
        "tiers": [
            {"range": (0, 10), "bonus": 0.08, "desc": "Very tight handle <10%"},
            {"range": (0, 12), "bonus": 0.04, "desc": "Acceptable handle <12%"},
        ],
    },
    {
        "factor": "Recovery (ideal >=70%)",
        "key": "recovery_pct",
        "tiers": [
            {"range": (70, 200), "bonus": 0.10, "desc": "Strong recovery >=70%"},
            {"range": (50, 70), "bonus": 0.05, "desc": "Moderate recovery >=50%"},
        ],
    },
    {
        "factor": "Prior Uptrend (ideal >=40%)",
        "key": "prior_advance_pct",
        "tiers": [
            {"range": (40, 500), "bonus": 0.10, "desc": "Strong prior advance >=40%"},
            {"range": (25, 40), "bonus": 0.05, "desc": "Adequate prior advance >=25%"},
        ],
    },
]

DOUBLE_BOTTOM_CRITERIA = [
    {
        "factor": "Depth (ideal 15-25%)",
        "key": "depth_pct",
        "tiers": [
            {"range": (15, 25), "bonus": 0.15, "desc": "Ideal depth range"},
            {"range": (12, 35), "bonus": 0.08, "desc": "Acceptable depth range"},
        ],
    },
    {
        "factor": "Undercut (ideal 1-3%)",
        "key": "undercut_pct",
        "tiers": [
            {"range": (1, 3), "bonus": 0.15, "desc": "Classic shakeout undercut"},
            {"range": (0.5, 5), "bonus": 0.08, "desc": "Valid undercut range"},
        ],
    },
    {
        "factor": "Mid-Peak Rise (ideal >=15%)",
        "key": "mid_peak_rise_pct",
        "tiers": [
            {"range": (15, 100), "bonus": 0.10, "desc": "Strong mid-peak rally >=15%"},
            {"range": (10, 15), "bonus": 0.05, "desc": "Adequate mid-peak rally >=10%"},
        ],
    },
    {
        "factor": "Duration (ideal 5-15 weeks)",
        "key": "duration_weeks",
        "tiers": [
            {"range": (5, 15), "bonus": 0.10, "desc": "Typical duration range"},
        ],
    },
    {
        "factor": "Prior Uptrend (ideal >=40%)",
        "key": "prior_advance_pct",
        "tiers": [
            {"range": (40, 500), "bonus": 0.10, "desc": "Strong prior advance >=40%"},
            {"range": (25, 40), "bonus": 0.05, "desc": "Adequate prior advance >=25%"},
        ],
    },
]

FLAT_BASE_CRITERIA = [
    {
        "factor": "Range (ideal <10%)",
        "key": "depth_pct",
        "tiers": [
            {"range": (0, 10), "bonus": 0.15, "desc": "Very tight range <10%"},
            {"range": (0, 15), "bonus": 0.08, "desc": "Acceptable range <15%"},
        ],
    },
    {
        "factor": "Duration (ideal 5-8 weeks)",
        "key": "duration_weeks",
        "tiers": [
            {"range": (5, 8), "bonus": 0.10, "desc": "Ideal duration 5-8 weeks"},
            {"range": (8, 52), "bonus": 0.05, "desc": "Longer consolidation >8 weeks"},
        ],
    },
    {
        "factor": "Tightness Score",
        "key": "tightness_score",
        "multiplier": 0.15,
        "desc": "Weekly close consistency",
    },
    {
        "factor": "Prior Uptrend (ideal >=40%)",
        "key": "prior_advance_pct",
        "tiers": [
            {"range": (40, 500), "bonus": 0.10, "desc": "Strong prior advance >=40%"},
            {"range": (25, 40), "bonus": 0.05, "desc": "Adequate prior advance >=25%"},
        ],
    },
]

# Map pattern types to their criteria
PATTERN_CRITERIA = {
    "cup_with_handle": CUP_WITH_HANDLE_CRITERIA,
    "cup_without_handle": CUP_WITH_HANDLE_CRITERIA,
    "double_bottom": DOUBLE_BOTTOM_CRITERIA,
    "flat_base": FLAT_BASE_CRITERIA,
}


def _check_tier(value: float, tier: dict) -> bool:
    """Check if a value falls within a tier's range."""
    low, high = tier["range"]
    return low <= value <= high


def _format_value(key: str, value: float) -> str:
    """Format a value for display based on its type."""
    if value is None:
        return "N/A"
    if key.endswith("_pct") or key == "tightness_score":
        return f"{value:.1f}%"
    if key.endswith("_weeks"):
        return f"{value:.1f} weeks"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return f"{value:.2f}"


def generate_explanation_factors(
    pattern_type: str, metadata: dict
) -> list[dict]:
    """Generate structured explanation of confidence factors.

    Args:
        pattern_type: The type of pattern (cup_with_handle, double_bottom, flat_base).
        metadata: The pattern metadata dict from the detector.

    Returns:
        List of dicts with keys:
        - factor: Factor name
        - value: Formatted value string
        - bonus: Points added to confidence
        - met: Whether the factor criteria was met
        - description: Explanation of what this means
    """
    if not metadata:
        return []

    criteria = PATTERN_CRITERIA.get(pattern_type, [])
    if not criteria:
        return []

    factors = []
    for criterion in criteria:
        key = criterion["key"]
        value = metadata.get(key)

        # Skip if value not in metadata
        if value is None:
            continue

        # Check if this criterion has a condition
        if "condition" in criterion:
            condition_key = criterion["condition"]
            if not metadata.get(condition_key):
                continue

        factor_info = {
            "factor": criterion["factor"],
            "value": _format_value(key, value),
            "bonus": 0.0,
            "met": False,
            "description": "",
        }

        # Handle boolean criteria
        if criterion.get("boolean"):
            if value:
                factor_info["bonus"] = criterion["bonus"]
                factor_info["met"] = True
                factor_info["description"] = criterion["desc"]
            else:
                factor_info["description"] = "Not present"
            factors.append(factor_info)
            continue

        # Handle multiplier criteria (like tightness_score)
        if "multiplier" in criterion:
            bonus = value * criterion["multiplier"]
            factor_info["bonus"] = bonus
            factor_info["met"] = bonus > 0
            factor_info["description"] = criterion["desc"]
            factors.append(factor_info)
            continue

        # Handle tiered criteria
        if "tiers" in criterion:
            for tier in criterion["tiers"]:
                if _check_tier(value, tier):
                    factor_info["bonus"] = tier["bonus"]
                    factor_info["met"] = True
                    factor_info["description"] = tier["desc"]
                    break
            else:
                factor_info["description"] = "Outside ideal range"

        factors.append(factor_info)

    return factors


def generate_narrative(
    pattern_type: str, confidence: float, metadata: dict
) -> str:
    """Generate brief narrative description of why pattern fits.

    Args:
        pattern_type: The type of pattern.
        confidence: The confidence score (0-1).
        metadata: The pattern metadata dict.

    Returns:
        A 2-3 sentence paragraph explaining the pattern characteristics.
    """
    if not metadata:
        return "No pattern metadata available for analysis."

    # Get the contributing factors
    factors = generate_explanation_factors(pattern_type, metadata)
    met_factors = [f for f in factors if f["met"]]
    unmet_factors = [f for f in factors if not f["met"] and f["bonus"] == 0]

    # Pattern-specific narratives
    if pattern_type in ("cup_with_handle", "cup_without_handle"):
        return _narrative_cup(pattern_type, confidence, metadata, met_factors)
    elif pattern_type == "double_bottom":
        return _narrative_double_bottom(confidence, metadata, met_factors)
    elif pattern_type == "flat_base":
        return _narrative_flat_base(confidence, metadata, met_factors)
    else:
        return f"Unclassified pattern with {confidence*100:.0f}% confidence."


def _narrative_cup(
    pattern_type: str,
    confidence: float,
    metadata: dict,
    met_factors: list[dict],
) -> str:
    """Generate narrative for cup patterns."""
    depth = metadata.get("depth_pct", 0)
    duration = metadata.get("duration_weeks", 0)
    has_handle = metadata.get("has_handle", False)
    handle_depth = metadata.get("handle_depth_pct", 0)
    recovery = metadata.get("recovery_pct", 0)
    prior_advance = metadata.get("prior_advance_pct", 0)

    # Build the narrative
    parts = []

    # Pattern type and quality assessment
    quality = _confidence_quality(confidence)
    pattern_name = "cup-with-handle" if has_handle else "cup-without-handle"
    parts.append(f"This {pattern_name} shows {quality} characteristics:")

    # Describe the cup formation
    cup_desc = f"a {depth:.0f}% decline over {duration:.0f} weeks"
    if recovery >= 70:
        cup_desc += " with strong recovery to the upper part of the base"
    elif recovery >= 50:
        cup_desc += " with recovery to the mid-range of the base"
    parts.append(cup_desc + ".")

    # Handle description
    if has_handle:
        handle_quality = "very tight" if handle_depth < 10 else "acceptable"
        parts.append(
            f"The handle pulled back {handle_depth:.1f}% ({handle_quality}) before the breakout attempt."
        )

    # Prior advance
    if prior_advance >= 40:
        parts.append(
            f"This follows a strong {prior_advance:.0f}% prior advance, confirming it's a continuation setup."
        )
    elif prior_advance >= 25:
        parts.append(
            f"This follows a {prior_advance:.0f}% prior advance."
        )

    return " ".join(parts)


def _narrative_double_bottom(
    confidence: float, metadata: dict, met_factors: list[dict]
) -> str:
    """Generate narrative for double bottom patterns."""
    depth = metadata.get("depth_pct", 0)
    duration = metadata.get("duration_weeks", 0)
    undercut = metadata.get("undercut_pct", 0)
    mid_peak_rise = metadata.get("mid_peak_rise_pct", 0)
    prior_advance = metadata.get("prior_advance_pct", 0)

    parts = []

    quality = _confidence_quality(confidence)
    parts.append(f"This double-bottom (W-shape) shows {quality} characteristics:")

    # Describe the formation
    parts.append(
        f"a {depth:.0f}% correction over {duration:.0f} weeks with a "
        f"{undercut:.1f}% undercut of the first low."
    )

    # Undercut quality
    if 1 <= undercut <= 3:
        parts.append(
            "The undercut is in the ideal 1-3% range for shaking out weak holders."
        )

    # Mid-peak
    if mid_peak_rise >= 15:
        parts.append(f"The mid-peak rallied {mid_peak_rise:.0f}% above the lows, showing strong demand.")
    elif mid_peak_rise >= 10:
        parts.append(f"The mid-peak rallied {mid_peak_rise:.0f}% above the lows.")

    # Prior advance
    if prior_advance >= 40:
        parts.append(f"The {prior_advance:.0f}% prior advance confirms this as a continuation pattern.")
    elif prior_advance >= 25:
        parts.append(f"Preceded by a {prior_advance:.0f}% advance.")

    return " ".join(parts)


def _narrative_flat_base(
    confidence: float, metadata: dict, met_factors: list[dict]
) -> str:
    """Generate narrative for flat base patterns."""
    depth = metadata.get("depth_pct", 0)
    duration = metadata.get("duration_weeks", 0)
    tightness = metadata.get("tightness_score", 0)
    prior_advance = metadata.get("prior_advance_pct", 0)

    parts = []

    quality = _confidence_quality(confidence)
    parts.append(f"This flat base shows {quality} characteristics:")

    # Describe the consolidation
    tightness_desc = "very tight" if depth < 10 else "reasonably tight"
    parts.append(
        f"a {tightness_desc} {depth:.0f}% consolidation over {duration:.0f} weeks."
    )

    # Tightness score
    if tightness >= 0.7:
        parts.append("Weekly closes are exceptionally consistent, indicating institutional accumulation.")
    elif tightness >= 0.5:
        parts.append("Weekly closes show good consistency.")

    # Prior advance
    if prior_advance >= 40:
        parts.append(
            f"This follows a powerful {prior_advance:.0f}% advance, making it a "
            "classic continuation pattern."
        )
    elif prior_advance >= 25:
        parts.append(f"This follows a {prior_advance:.0f}% advance.")

    return " ".join(parts)


def _confidence_quality(confidence: float) -> str:
    """Convert confidence score to quality description."""
    if confidence >= 0.85:
        return "excellent"
    elif confidence >= 0.75:
        return "strong"
    elif confidence >= 0.65:
        return "good"
    elif confidence >= 0.55:
        return "moderate"
    else:
        return "weak"
