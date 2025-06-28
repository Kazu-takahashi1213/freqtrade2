
from __future__ import annotations

import pandas as pd


def triple_barrier_label(
    df: pd.DataFrame,
    pt: float = 0.02,
    sl: float = 0.02,
    max_periods: int = 12,
    price_col: str = "close",
) -> pd.Series:
    """Generate labels using the triple barrier method.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with OHLC data.
    pt : float, default 0.02
        Profit-taking threshold as fraction of price.
    sl : float, default 0.02
        Stop-loss threshold as fraction of price.
    max_periods : int, default 12
        Number of periods to use as vertical barrier.
    price_col : str, default "close"
        Name of the price column to evaluate.

    Returns
    -------
    pd.Series
        Series of labels (1 for up, -1 for down, 0 for neutral).
    """
    prices = df[price_col]
    labels = pd.Series(index=df.index, dtype="float")

    for i in range(len(df)):
        if i + 1 >= len(df):
            break
        end = min(i + max_periods, len(df) - 1)
        horizon = prices.iloc[i + 1 : end + 1]
        entry_price = prices.iloc[i]
        up = entry_price * (1 + pt)
        down = entry_price * (1 - sl)
        label = 0
        for price in horizon:
            if price >= up:
                label = 1
                break
            if price <= down:
                label = -1
                break
        labels.iloc[i] = label

    labels.iloc[len(df) - max_periods :] = pd.NA
    return labels
