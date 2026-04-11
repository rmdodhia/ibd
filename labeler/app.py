"""Streamlit labeling UI for reviewing and correcting auto-labels.

Features:
- Paginated list of detected patterns
- Interactive price chart with pattern annotations
- Label controls (confirm, override, flag)
- Progress tracker

Usage:
    streamlit run labeler/app.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from labeler.queries import (
    get_patterns_paginated,
    get_pattern_with_features,
    update_pattern_label,
    update_pattern_type_override,
    get_progress_stats,
    get_disagreement_stats,
    get_distinct_symbols,
    get_distinct_pattern_types,
    get_price_data_for_chart,
)
from labeler.components.chart import create_pattern_chart
from scanner.patterns.explanations import (
    generate_explanation_factors,
    generate_narrative,
)

# Page config
st.set_page_config(
    page_title="IBD Pattern Labeler",
    page_icon="📈",
    layout="wide",
)

# Initialize session state
if "selected_pattern_id" not in st.session_state:
    st.session_state.selected_pattern_id = None
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "last_filters" not in st.session_state:
    st.session_state.last_filters = None


def render_sidebar():
    """Render the sidebar with progress tracker and filters."""
    st.sidebar.header("Labeling Progress")

    # Progress stats
    stats = get_progress_stats()

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Reviewed", stats["reviewed"])
    col2.metric("Total", stats["total"])

    if stats["total"] > 0:
        st.sidebar.progress(stats["pct_complete"] / 100)
        st.sidebar.caption(f"{stats['pct_complete']:.1f}% complete")

    if stats["reviewed"] > 0:
        st.sidebar.caption(f"Agreement rate: {stats['agreement_rate']:.1f}%")

        # Show disagreement analysis if enough samples
        disagree_stats = get_disagreement_stats()
        if not disagree_stats.get("insufficient_data") and disagree_stats.get("n_reviewed", 0) >= 10:
            with st.sidebar.expander("Label Strategy Comparison"):
                best_strategy = None
                best_rate = 0

                for col, data in disagree_stats.get("by_strategy", {}).items():
                    rate = data["rate"]
                    status = "green" if rate >= 80 else "orange" if rate >= 60 else "red"
                    st.markdown(
                        f":{status}[{data['name']}]: {rate:.0f}% ({data['agreed']}/{data['total']})"
                    )
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = data["name"]

                if best_strategy:
                    st.caption(f"Best match: {best_strategy}")

            # Disagreement breakdown
            dtypes = disagree_stats.get("disagreement_types", {})
            h_s_a_f = dtypes.get("human_success_auto_failure", 0)
            h_f_a_s = dtypes.get("human_failure_auto_success", 0)

            if h_s_a_f + h_f_a_s > 0:
                with st.sidebar.expander("Disagreement Types"):
                    if h_s_a_f > 0:
                        st.markdown(f"Human=success, Auto=failure: **{h_s_a_f}**")
                        if h_s_a_f > h_f_a_s:
                            st.caption("Auto-labels may be too strict")
                    if h_f_a_s > 0:
                        st.markdown(f"Human=failure, Auto=success: **{h_f_a_s}**")
                        if h_f_a_s > h_s_a_f:
                            st.caption("Auto-labels may be too loose")

            # Pattern type agreement
            by_ptype = disagree_stats.get("by_pattern_type", {})
            low_agreement = {k: v for k, v in by_ptype.items() if v["rate"] < 80}
            if low_agreement:
                with st.sidebar.expander("Patterns Needing Review"):
                    for ptype, data in low_agreement.items():
                        st.markdown(
                            f"**{ptype}**: {data['rate']:.0f}% "
                            f"({data['agreed']}/{data['total']})"
                        )

    st.sidebar.divider()

    # Filters
    st.sidebar.header("Filters")

    # Symbol filter
    symbols = ["All"] + get_distinct_symbols()
    selected_symbol = st.sidebar.selectbox(
        "Symbol",
        options=symbols,
        index=0,
    )

    # Pattern type filter
    pattern_types = ["All"] + get_distinct_pattern_types()
    selected_pattern_type = st.sidebar.selectbox(
        "Pattern Type",
        options=pattern_types,
        index=0,
    )

    # Outcome filter
    outcomes = ["All", "success", "failure", "pending"]
    selected_outcome = st.sidebar.selectbox(
        "Outcome",
        options=outcomes,
        index=0,
    )

    # Reviewed status filter
    reviewed_options = ["All", "Reviewed", "Unreviewed"]
    selected_reviewed = st.sidebar.selectbox(
        "Review Status",
        options=reviewed_options,
        index=0,
    )

    # Convert to query params
    filters = {
        "symbol": selected_symbol if selected_symbol != "All" else None,
        "pattern_type": selected_pattern_type if selected_pattern_type != "All" else None,
        "outcome": selected_outcome if selected_outcome != "All" else None,
        "reviewed": (
            True if selected_reviewed == "Reviewed"
            else False if selected_reviewed == "Unreviewed"
            else None
        ),
    }

    return filters


def _outcome_color(outcome: str) -> str:
    """Get color for outcome display."""
    if outcome == "success":
        return "green"
    elif outcome == "failure":
        return "red"
    elif outcome == "pending":
        return "orange"
    elif outcome == "neutral":
        return "gray"
    return "gray"


def render_pattern_details(pattern: dict):
    """Render pattern details in 4-column layout."""
    st.subheader("Pattern Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Symbol**")
        st.write(pattern["symbol"])
        st.markdown("**Pattern Type**")
        st.write(pattern["pattern_type"])
        st.markdown("**Pivot Date**")
        st.write(pattern["pivot_date"])

    with col2:
        st.markdown("**Pivot Price**")
        st.write(f"${pattern['pivot_price']:.2f}" if pattern["pivot_price"] else "N/A")
        st.markdown("**Base Period**")
        st.write(f"{pattern['base_start_date']} to {pattern['base_end_date']}")
        st.markdown("**Auto Label**")
        label_color = (
            "green" if pattern["auto_label"] == "success"
            else "red" if pattern["auto_label"] == "failure"
            else "gray"
        )
        st.markdown(f":{label_color}[{pattern['auto_label'] or 'N/A'}]")

    with col3:
        st.markdown("**Max Gain %**")
        gain = pattern["outcome_max_gain_pct"]
        st.write(f"{gain:.1f}%" if gain is not None else "N/A")
        st.markdown("**Max Loss %**")
        loss = pattern["outcome_max_loss_pct"]
        st.write(f"{loss:.1f}%" if loss is not None else "N/A")
        st.markdown("**Reviewed**")
        st.write("Yes" if pattern["reviewed"] else "No")

    with col4:
        st.markdown("**Human Label**")
        human_label = pattern["human_label"]
        if human_label:
            human_color = (
                "green" if human_label == "success"
                else "red" if human_label == "failure"
                else "yellow" if human_label == "ambiguous"
                else "gray"
            )
            st.markdown(f":{human_color}[{human_label}]")
        else:
            st.write("Not set")

    # Multi-label outcomes section
    st.divider()
    st.subheader("Label Strategies (12-week window)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**+20% / -7% (Original IBD)**")
        outcome = pattern.get("outcome_asym_20_7")
        ret = pattern.get("return_asym_20_7")
        if outcome:
            color = _outcome_color(outcome)
            st.markdown(f":{color}[{outcome}]")
            if ret is not None:
                st.write(f"Return: {ret:.1f}%")
        else:
            st.write("Not computed - run labeler")

    with col2:
        st.markdown("**+15% / -10% (Less Extreme)**")
        outcome = pattern.get("outcome_asym_15_10")
        ret = pattern.get("return_asym_15_10")
        if outcome:
            color = _outcome_color(outcome)
            st.markdown(f":{color}[{outcome}]")
            if ret is not None:
                st.write(f"Return: {ret:.1f}%")
        else:
            st.write("Not computed - run labeler")

    with col3:
        st.markdown("**+10% / -10% (Symmetric)**")
        outcome = pattern.get("outcome_sym_10")
        ret = pattern.get("return_sym_10")
        if outcome:
            color = _outcome_color(outcome)
            st.markdown(f":{color}[{outcome}]")
            if ret is not None:
                st.write(f"Return: {ret:.1f}%")
        else:
            st.write("Not computed - run labeler")


def render_confidence_section(pattern: dict):
    """Render pattern confidence display with explanation factors."""
    confidence = pattern.get("confidence")
    metadata = pattern.get("metadata")
    pattern_type = pattern.get("pattern_type")

    st.divider()
    st.subheader("Pattern Confidence")

    if confidence is None:
        st.info(
            "Confidence data not available for this pattern. "
            "Re-run the labeler with `--force` to compute confidence scores."
        )
        return

    # Confidence metric and progress bar
    col1, col2 = st.columns([1, 3])

    with col1:
        confidence_pct = confidence * 100
        if confidence_pct >= 75:
            delta_color = "normal"
        elif confidence_pct >= 60:
            delta_color = "off"
        else:
            delta_color = "inverse"
        st.metric(
            "Confidence",
            f"{confidence_pct:.0f}%",
            delta=None,
        )

    with col2:
        st.progress(confidence)

    # Narrative summary
    if metadata:
        narrative = generate_narrative(pattern_type, confidence, metadata)
        st.markdown(f"*{narrative}*")

        # Contributing factors in expander
        factors = generate_explanation_factors(pattern_type, metadata)
        if factors:
            with st.expander("Contributing Factors", expanded=False):
                for factor in factors:
                    # Color code based on whether factor was met
                    if factor["met"]:
                        icon = ":green[+]"
                        bonus_str = f"+{factor['bonus']*100:.0f}%"
                    else:
                        icon = ":gray[-]"
                        bonus_str = "+0%"

                    st.markdown(
                        f"{icon} **{factor['factor']}** ({factor['value']}): "
                        f"{factor['description']} -> {bonus_str}"
                    )
    else:
        st.caption("No metadata available for detailed explanation.")


def render_pattern_type_override(pattern: dict):
    """Render pattern type override controls."""
    st.divider()
    st.subheader("Pattern Type Verification")

    current_type = pattern.get("pattern_type", "unknown")
    current_override = pattern.get("pattern_type_override")

    # Display current state
    if current_override:
        st.markdown(
            f"**Detected:** {current_type} | "
            f"**Override:** :orange[{current_override}]"
        )
    else:
        st.markdown(f"**Detected:** {current_type} (no override)")

    # Override options
    override_options = [
        ("Correct (accept auto-detected)", None),
        ("Not a valid pattern", "not_a_pattern"),
        ("Actually: cup_with_handle", "cup_with_handle"),
        ("Actually: cup_without_handle", "cup_without_handle"),
        ("Actually: double_bottom", "double_bottom"),
        ("Actually: flat_base", "flat_base"),
    ]

    # Find current selection index
    current_selection = 0
    for i, (label, value) in enumerate(override_options):
        if value == current_override:
            current_selection = i
            break

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_label = st.selectbox(
            "Pattern Type Override",
            options=[opt[0] for opt in override_options],
            index=current_selection,
            label_visibility="collapsed",
        )

    with col2:
        # Find the value for selected label
        selected_value = None
        for label, value in override_options:
            if label == selected_label:
                selected_value = value
                break

        if st.button("Save Override", use_container_width=True):
            if update_pattern_type_override(pattern["id"], selected_value):
                if selected_value:
                    st.success(f"Override set to '{selected_value}'")
                else:
                    st.success("Override cleared")
                st.rerun()
            else:
                st.error("Failed to update override")


def render_features(features: dict):
    """Render pattern features in an expandable section."""
    with st.expander("Pattern Features", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Base Metrics**")
            st.write(f"Depth: {features.get('base_depth_pct', 'N/A'):.1f}%" if features.get('base_depth_pct') else "Depth: N/A")
            st.write(f"Duration: {features.get('base_duration_weeks', 'N/A'):.1f} weeks" if features.get('base_duration_weeks') else "Duration: N/A")
            st.write(f"Symmetry: {features.get('base_symmetry', 'N/A'):.2f}" if features.get('base_symmetry') else "Symmetry: N/A")
            if features.get('handle_depth_pct') is not None:
                st.write(f"Handle Depth: {features['handle_depth_pct']:.1f}%")

        with col2:
            st.markdown("**Volume Metrics**")
            st.write(f"Breakout Vol Ratio: {features.get('breakout_volume_ratio', 'N/A'):.2f}" if features.get('breakout_volume_ratio') else "Breakout Vol Ratio: N/A")
            st.write(f"Vol Trend: {features.get('volume_trend_in_base', 'N/A'):.2f}" if features.get('volume_trend_in_base') else "Vol Trend: N/A")
            st.write(f"Up/Down Vol: {features.get('up_down_volume_ratio', 'N/A'):.2f}" if features.get('up_down_volume_ratio') else "Up/Down Vol: N/A")
            st.write(f"Tightness: {features.get('tightness_score', 'N/A'):.2f}" if features.get('tightness_score') else "Tightness: N/A")

        with col3:
            st.markdown("**Relative Strength**")
            st.write(f"RS 4wk Slope: {features.get('rs_line_slope_4wk', 'N/A'):.4f}" if features.get('rs_line_slope_4wk') else "RS 4wk Slope: N/A")
            st.write(f"RS 12wk Slope: {features.get('rs_line_slope_12wk', 'N/A'):.4f}" if features.get('rs_line_slope_12wk') else "RS 12wk Slope: N/A")
            st.write(f"RS New High: {'Yes' if features.get('rs_new_high') else 'No'}")
            st.write(f"RS Rank: {features.get('rs_rank_percentile', 'N/A'):.0f}%" if features.get('rs_rank_percentile') else "RS Rank: N/A")

        with col4:
            st.markdown("**Fundamentals**")
            st.write(f"EPS YoY: {features.get('eps_latest_yoy_growth', 'N/A'):.1f}%" if features.get('eps_latest_yoy_growth') else "EPS YoY: N/A")
            st.write(f"EPS Accel: {features.get('eps_acceleration', 'N/A'):.1f}%" if features.get('eps_acceleration') else "EPS Accel: N/A")
            st.write(f"Rev YoY: {features.get('revenue_latest_yoy_growth', 'N/A'):.1f}%" if features.get('revenue_latest_yoy_growth') else "Rev YoY: N/A")
            st.write(f"Inst %: {features.get('institutional_pct', 'N/A'):.1f}%" if features.get('institutional_pct') else "Inst %: N/A")


def render_label_controls(pattern: dict):
    """Render label controls for confirm/override/flag."""
    st.subheader("Label Controls")

    col1, col2, col3, col4 = st.columns(4)

    current_label = pattern.get("human_label") or pattern.get("auto_label")

    with col1:
        if st.button(
            "Confirm Auto-Label",
            type="primary" if pattern["auto_label"] else "secondary",
            disabled=not pattern["auto_label"],
            use_container_width=True,
        ):
            if update_pattern_label(pattern["id"], pattern["auto_label"], reviewed=True):
                st.success(f"Confirmed as '{pattern['auto_label']}'")
                st.rerun()
            else:
                st.error("Failed to update label")

    with col2:
        if st.button(
            "Mark Success",
            type="primary" if current_label == "success" else "secondary",
            use_container_width=True,
        ):
            if update_pattern_label(pattern["id"], "success", reviewed=True):
                st.success("Marked as 'success'")
                st.rerun()
            else:
                st.error("Failed to update label")

    with col3:
        if st.button(
            "Mark Failure",
            type="primary" if current_label == "failure" else "secondary",
            use_container_width=True,
        ):
            if update_pattern_label(pattern["id"], "failure", reviewed=True):
                st.success("Marked as 'failure'")
                st.rerun()
            else:
                st.error("Failed to update label")

    with col4:
        if st.button(
            "Flag Ambiguous",
            type="secondary",
            use_container_width=True,
        ):
            if update_pattern_label(pattern["id"], "ambiguous", reviewed=True):
                st.warning("Flagged as 'ambiguous'")
                st.rerun()
            else:
                st.error("Failed to update label")


def render_pattern_list(patterns: list[dict], total_count: int, page: int, page_size: int):
    """Render paginated pattern list with selection."""
    st.subheader(f"Patterns ({total_count} total)")

    if not patterns:
        st.info("No patterns found matching the filters.")
        return

    # Create DataFrame for display
    display_data = []
    for p in patterns:
        display_data.append({
            "ID": p["id"],
            "Symbol": p["symbol"],
            "Type": p["pattern_type"],
            "Pivot Date": p["pivot_date"],
            "Outcome": p["outcome"] or "pending",
            "Return %": f"{p['outcome_return_pct']:.1f}%" if p["outcome_return_pct"] is not None else "N/A",
            "Auto Label": p["auto_label"] or "N/A",
            "Human Label": p["human_label"] or "-",
            "Reviewed": "Yes" if p["reviewed"] else "No",
        })

    # Display as selectable dataframe
    selected_idx = None
    for i, p in enumerate(patterns):
        if p["id"] == st.session_state.selected_pattern_id:
            selected_idx = i
            break

    # Use data editor for selection
    selection = st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Handle selection
    if selection and selection.selection and selection.selection.rows:
        selected_row = selection.selection.rows[0]
        new_id = patterns[selected_row]["id"]
        if new_id != st.session_state.selected_pattern_id:
            st.session_state.selected_pattern_id = new_id
            st.rerun()

    # Pagination controls
    total_pages = (total_count + page_size - 1) // page_size
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("Previous", disabled=page <= 1):
                st.session_state.current_page = page - 1
                st.session_state.selected_pattern_id = None
                st.rerun()

        with col2:
            st.markdown(f"<center>Page {page} of {total_pages}</center>", unsafe_allow_html=True)

        with col3:
            if st.button("Next", disabled=page >= total_pages):
                st.session_state.current_page = page + 1
                st.session_state.selected_pattern_id = None
                st.rerun()


def main():
    """Main app entry point."""
    st.title("IBD Pattern Labeler")

    # Sidebar: progress and filters
    filters = render_sidebar()

    # Reset pagination/selection when any filter changes so the chart and
    # details always reflect the filtered result set.
    if filters != st.session_state.last_filters:
        st.session_state.current_page = 1
        st.session_state.selected_pattern_id = None
        st.session_state.last_filters = filters.copy()

    # Get patterns for current page
    page_size = 15
    patterns, total_count = get_patterns_paginated(
        page=st.session_state.current_page,
        page_size=page_size,
        **filters,
    )

    # Reset page if filters changed and no results
    if not patterns and st.session_state.current_page > 1:
        st.session_state.current_page = 1
        st.session_state.selected_pattern_id = None
        st.rerun()

    # Auto-select first pattern if none selected
    if patterns and st.session_state.selected_pattern_id is None:
        st.session_state.selected_pattern_id = patterns[0]["id"]

    # Main content area
    if st.session_state.selected_pattern_id:
        pattern = get_pattern_with_features(st.session_state.selected_pattern_id)

        if pattern:
            # Chart
            price_data = get_price_data_for_chart(
                symbol=pattern["symbol"],
                base_start_date=pattern["base_start_date"],
                pivot_date=pattern["pivot_date"],
                days_before=60,
                days_after=60,
            )

            fig = create_pattern_chart(price_data, pattern)
            st.plotly_chart(fig, use_container_width=True)

            # Pattern details
            render_pattern_details(pattern)

            # Confidence and explanation
            render_confidence_section(pattern)

            # Pattern type override
            render_pattern_type_override(pattern)

            # Features
            render_features(pattern["features"])

            # Label controls
            render_label_controls(pattern)

            st.divider()
        else:
            st.warning("Selected pattern not found.")
            st.session_state.selected_pattern_id = None

    # Pattern list at bottom
    render_pattern_list(
        patterns,
        total_count,
        st.session_state.current_page,
        page_size,
    )


if __name__ == "__main__":
    main()
