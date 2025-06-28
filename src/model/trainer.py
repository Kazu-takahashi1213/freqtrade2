"""Model training utilities using LightGBM."""

from __future__ import annotations

import os
from typing import Tuple

import lightgbm as lgb
import pandas as pd

from src.data_pipeline.feature_store import compute_features
from src.data_pipeline.labeling import triple_barrier_label


def prepare_dataset(
    df: pd.DataFrame,
    pt_sl: Tuple[float, float] = (0.02, 0.02),
    max_periods: int = 12,
) -> pd.DataFrame:
    """Return dataframe with features and triple barrier labels."""
    feat_df = compute_features(df)
    labels = triple_barrier_label(
        feat_df, pt=pt_sl[0], sl=pt_sl[1], max_periods=max_periods
    )
    # map -1,0,1 -> 0,1,2 for LightGBM multiclass
    mapping = {-1: 0, 0: 1, 1: 2}
    feat_df["direction"] = labels.map(mapping)
    return feat_df.dropna()


def train_model(
    df: pd.DataFrame,
    target_col: str = "direction",
    model_path: str | None = None,
) -> lgb.Booster:
    """Train LightGBM model on provided dataframe and optionally save it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and target column.
    target_col : str, default 'direction'
        Name of the target column.

    Returns
    -------
    lgb.Booster
        Trained LightGBM model.
    """
    features = df.drop(columns=[target_col])
    target = df[target_col]

    train_size = int(len(df) * 0.8)
    train_set = lgb.Dataset(features.iloc[:train_size], label=target.iloc[:train_size])
    valid_set = lgb.Dataset(features.iloc[train_size:], label=target.iloc[train_size:])
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
    }
    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    if model_path:
        model.save_model(model_path)

    return model


def load_model(model_path: str) -> lgb.Booster | None:
    """Load a LightGBM model from path if it exists."""
    if os.path.exists(model_path):
        return lgb.Booster(model_file=model_path)
    return None