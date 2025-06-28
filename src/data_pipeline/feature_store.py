
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def _ma_deviation(series: pd.Series, window: int = 20) -> pd.Series:
    """Return deviation from moving average."""
    ma = series.rolling(window=window).mean()
    return (series - ma) / ma


def compute_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain columns: 'open', 'high', 'low', 'close', 'volume'.

    Returns
    -------
    pd.DataFrame
        Input dataframe with indicator columns appended.
    """
    df = dataframe.copy()

    # MACD
    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)

    # RSI
    df["rsi"] = ta.rsi(df["close"])

    # Bollinger Bands
    bbands = ta.bbands(df["close"])
    df = pd.concat([df, bbands], axis=1)

    # ATR
    df["atr"] = ta.atr(df["high"], df["low"], df["close"])

    # Stochastic oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)

    # Commodity Channel Index
    df["cci"] = ta.cci(df["high"], df["low"], df["close"])

    # Money Flow Index
    df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])

    # Deviation from 20-period moving average
    df["ma_dev"] = _ma_deviation(df["close"], window=20)

    # Orderbook depth ratio if data available
    if {"bid_volume", "ask_volume"} <= set(df.columns):
        depth_sum = df["bid_volume"] + df["ask_volume"]
        depth_sum = depth_sum.replace(0, pd.NA)
        df["depth_ratio"] = (df["bid_volume"] - df["ask_volume"]) / depth_sum

    return df

