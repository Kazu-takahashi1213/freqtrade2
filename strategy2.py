
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)


import talib.abstract as ta
from technical import qtpylib


class Strategy2(IStrategy):

    can_short: bool = False

    minimal_roi = {
        
        "0": 0.03,
        "30": 0.02,
        "60": 0.01,
        "120": 0.00
    }

    stoploss = -0.02

    trailing_stop = False

    timeframe = "5m"
 
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    startup_candle_count: int = 200
 
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        
        from src.model.trainer import load_model
        self.model = load_model(self.model_path)

    def informative_pairs(self):
        
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["adx"] = ta.ADX(dataframe)

        dataframe["rsi"] = ta.RSI(dataframe)

        dataframe["mfi"] = ta.MFI(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]

        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["enter_long"]  = 0
        dataframe["enter_short"] = 0

        features = dataframe[self.feature_columns].fillna(0)
        preds    = self.model.predict(features)
        signal   = preds.argmax(axis=1)

    
        cond_trend_up = (
            (dataframe["tema"] > dataframe["tema"].shift(1))      # TEMA上昇
            & (dataframe["adx"]  > 25)                            # ADXが25超（強トレンド）
        )
        cond_momo = (
            (dataframe["rsi"]  < 30)                              # RSIが30以下（反転余地あり）
            & (dataframe["mfi"] > 50)                             # MFIが50超（資金流入あり）
        )
        cond_vola = (
            dataframe["bb_width"] < 0.05                          # BB幅が0.05以下（スクイーズ直後）
        )


        long_signal = (signal == 2) & cond_trend_up & cond_momo & cond_vola
        dataframe.loc[long_signal, "enter_long"] = 1

        short_signal = (signal == 0) & cond_trend_up & cond_momo & cond_vola
        dataframe.loc[short_signal, "enter_short"] = 1

    
        rr_ratio = self.minimal_roi["0"] / abs(self.stoploss)
        if rr_ratio < 1.5:
            dataframe.loc[:, ["enter_long", "enter_short"]] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0

    
        cond_exit_long = (
            (qtpylib.crossed_above(dataframe["rsi"], 70))
            & (dataframe["tema"] < dataframe["tema"].shift(1))
            & (dataframe["bb_width"] > 0.05)
        )
        dataframe.loc[cond_exit_long, "exit_long"] = 1

   
        cond_exit_short = (
            (qtpylib.crossed_below(dataframe["rsi"], 30))
            & (dataframe["tema"] > dataframe["tema"].shift(1))
            & (dataframe["bb_width"] > 0.05)
        )
        dataframe.loc[cond_exit_short, "exit_short"] = 1

        return dataframe