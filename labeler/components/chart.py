"""Chart component for pattern visualization.

Creates interactive candlestick charts with pattern annotations.
"""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner.config import get


def create_pattern_chart(
    df: pd.DataFrame,
    pattern: dict,
    height: int = 600,
) -> go.Figure:
    """Create an interactive candlestick chart with pattern annotations.

    Args:
        df: DataFrame with columns [date, open, high, low, close, volume].
        pattern: Pattern dict with base_start_date, base_end_date, pivot_date, pivot_price.
        height: Chart height in pixels.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No price data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        return fig

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    # Calculate moving averages
    df = df.copy()
    df["ma50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["ma200"] = df["close"].rolling(window=200, min_periods=1).mean()

    # Create subplots: candlestick + volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    # 50-day MA
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma50"],
            name="50 MA",
            line=dict(color="#ff9800", width=1.5),
        ),
        row=1,
        col=1,
    )

    # 200-day MA
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ma200"],
            name="200 MA",
            line=dict(color="#2196f3", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Volume bars
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # Pattern annotations
    base_start = pd.to_datetime(pattern.get("base_start_date"))
    base_end = pd.to_datetime(pattern.get("base_end_date"))
    pivot_date = pd.to_datetime(pattern.get("pivot_date"))
    pivot_price = pattern.get("pivot_price")

    # Get y-axis range for annotations
    y_min = df["low"].min()
    y_max = df["high"].max()
    y_range = y_max - y_min

    # Blue shaded rectangle for base period
    if pd.notna(base_start) and pd.notna(base_end):
        fig.add_vrect(
            x0=base_start,
            x1=base_end,
            fillcolor="rgba(33, 150, 243, 0.15)",
            layer="below",
            line_width=0,
            annotation_text="Base",
            annotation_position="top left",
            row=1,
            col=1,
        )

    # Vertical line at pivot date
    # Use add_shape + add_annotation instead of add_vline because recent
    # pandas versions no longer allow Plotly's internal Timestamp + int math.
    if pd.notna(pivot_date):
        fig.add_shape(
            type="line",
            x0=pivot_date,
            x1=pivot_date,
            y0=y_min - 0.05 * y_range,
            y1=y_max + 0.05 * y_range,
            xref="x",
            yref="y",
            line=dict(color="purple", width=2, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=pivot_date,
            y=y_max + 0.03 * y_range,
            xref="x",
            yref="y",
            text="Pivot",
            showarrow=False,
            font=dict(color="purple"),
        )

    # Success and failure threshold lines
    if pivot_price:
        min_gain_pct = get("breakout.min_gain_pct", 20)
        max_loss_pct = get("breakout.max_loss_pct", 7)

        success_price = pivot_price * (1 + min_gain_pct / 100)
        failure_price = pivot_price * (1 - max_loss_pct / 100)

        # Success threshold (+20%)
        fig.add_hline(
            y=success_price,
            line_dash="dot",
            line_color="green",
            line_width=1.5,
            annotation_text=f"+{min_gain_pct}%",
            annotation_position="right",
            row=1,
            col=1,
        )

        # Failure threshold (-7%)
        fig.add_hline(
            y=failure_price,
            line_dash="dot",
            line_color="red",
            line_width=1.5,
            annotation_text=f"-{max_loss_pct}%",
            annotation_position="right",
            row=1,
            col=1,
        )

        # Pivot price line
        fig.add_hline(
            y=pivot_price,
            line_dash="solid",
            line_color="purple",
            line_width=1,
            annotation_text=f"Pivot ${pivot_price:.2f}",
            annotation_position="left",
            row=1,
            col=1,
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{pattern.get('symbol', '')} - {pattern.get('pattern_type', '')}",
            font=dict(size=16),
        ),
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=40),
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Update x-axes
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
        ],
    )

    return fig
