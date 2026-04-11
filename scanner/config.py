"""Load and access configuration from config.yaml."""

import yaml
from pathlib import Path
from typing import Any

_config: dict | None = None


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Configuration dictionary.
    """
    global _config
    if _config is not None and path is None:
        return _config

    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"

    with open(path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def get(key_path: str, default: Any = None) -> Any:
    """Get a config value by dot-separated path.

    Args:
        key_path: Dot-separated path like 'patterns.cup_with_handle.min_depth_pct'
        default: Default value if key not found.

    Returns:
        Configuration value.

    Example:
        >>> get('breakout.min_gain_pct')
        20
    """
    config = load_config()
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def get_price_range(df: "pd.DataFrame", use_intraday: bool | None = None) -> tuple:
    """Return (high_prices, low_prices) based on config.

    If use_intraday is True: returns (df["high"], df["low"])
    If use_intraday is False: returns (df["close"], df["close"])

    Args:
        df: DataFrame with columns [high, low, close].
        use_intraday: Override config setting. If None, reads from config.

    Returns:
        Tuple of (high_prices, low_prices) Series.
    """
    if use_intraday is None:
        use_intraday = get("breakout.use_intraday_prices", True)

    if use_intraday:
        return df["high"], df["low"]
    else:
        return df["close"], df["close"]


def get_price_high_low_arrays(df: "pd.DataFrame", use_intraday: bool | None = None) -> tuple:
    """Return (high_array, low_array) as numpy arrays based on config.

    If use_intraday is True: returns (df["high"].values, df["low"].values)
    If use_intraday is False: returns (df["close"].values, df["close"].values)

    Args:
        df: DataFrame with columns [high, low, close].
        use_intraday: Override config setting. If None, reads from config.

    Returns:
        Tuple of (high_array, low_array) numpy arrays.
    """
    if use_intraday is None:
        use_intraday = get("breakout.use_intraday_prices", True)

    if use_intraday:
        return df["high"].values, df["low"].values
    else:
        return df["close"].values, df["close"].values
