from __future__ import annotations

from typing import Any, Dict, List
import os

import pandas as pd
from freqtrade.strategy.interface import IStrategy

from src.data_pipeline.feature_store import compute_features
from src.model.trainer import load_model, train_model, prepare_dataset


class ShortTermStrategy(IStrategy):
    minimal_roi = {"0": 0.03,
                   "30": 0.02,
                   "60": 0.01,
                   "120": 0.00}
    stoploss = -0.02
    timeframe = "5m"

    max_hold_bars = 48

    model: Any = None
    model_path: str = os.path.join("models", "lightgbm.txt")
    feature_columns: List[str] | None = None

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.model = load_model(self.model_path)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Add technical indicator features."""
        df = compute_features(dataframe)

        if self.feature_columns is None:
            self.feature_columns = [c for c in df.columns if c not in {"date", "direction"}]

        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Generate entry signals using the trained model."""
        if self.model is None:
            self.model = load_model(self.model_path)

        if self.model is None and "direction" in dataframe.columns:
            self.model = train_model(dataframe.dropna(), model_path=self.model_path)
        elif self.model is None:
            train_df = prepare_dataset(dataframe)
            self.model = train_model(train_df, model_path=self.model_path)

        if self.model and self.feature_columns:
            features = dataframe[self.feature_columns].fillna(0)
            preds = self.model.predict(features)
            signal = preds.argmax(axis=1)
            dataframe.loc[signal == 2, "enter_long"] = 1
            dataframe.loc[signal == 0, "enter_short"] = 1

        if self.feature_columns:
            sl = abs(self.stoploss)
            rr_ratio = self.minimal_roi["0"] / sl
            if rr_ratio < 1.5:
                dataframe.loc[:, "enter_long"] = 0
                dataframe.loc[:, "enter_short"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """Simple exit conditions based on price move."""
        dataframe["exit_long"] = dataframe["close"] > dataframe["close"].shift(1)
        dataframe["exit_short"] = dataframe["close"] < dataframe["close"].shift(1)
        return dataframe
    