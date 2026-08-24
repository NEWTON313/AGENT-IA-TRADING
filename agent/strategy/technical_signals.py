"""
Moteur de signaux techniques — implémentation unique de la stratégie validée
(voir RESEARCH_SUMMARY.md) : Supertrend(10,3) daily + filtre de tendance
Weekly EMA10, sans lookahead. Remplace les implémentations dupliquées dans
les scripts de recherche 09/10/15/16/17.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from agent import config


def supertrend(df: pd.DataFrame, atr_period: int = config.SUPERTREND_ATR_PERIOD,
                multiplier: float = config.SUPERTREND_MULTIPLIER) -> pd.Series:
    """
    Calcule la direction Supertrend (+1 haussier / -1 baissier) pour chaque bougie.
    """
    high, low, close = df["high"], df["low"], df["close"]
    atr = vbt.ATR.run(high, low, close, window=atr_period).atr
    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = np.ones(len(close), dtype=int)

    close_v = close.values
    fu = final_upper.values.copy()
    fl = final_lower.values.copy()

    for i in range(1, len(close_v)):
        if close_v[i] > fu[i - 1]:
            direction[i] = 1
        elif close_v[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1 and fl[i] < fl[i - 1]:
                fl[i] = fl[i - 1]
            if direction[i] == -1 and fu[i] > fu[i - 1]:
                fu[i] = fu[i - 1]

    return pd.Series(direction, index=close.index, name="supertrend_direction")


def supertrend_crosses(df: pd.DataFrame, **kwargs) -> tuple[pd.Series, pd.Series]:
    """Retourne (bull_cross, bear_cross) : True le jour où Supertrend bascule."""
    direction = supertrend(df, **kwargs)
    bull_cross = (direction == 1) & (direction.shift(1) == -1)
    bear_cross = (direction == -1) & (direction.shift(1) == 1)
    return bull_cross.fillna(False), bear_cross.fillna(False)


def weekly_trend_filter(daily_index: pd.DatetimeIndex, weekly_df: pd.DataFrame,
                         ema_window: int = config.WEEKLY_TREND_EMA_WINDOW) -> pd.Series:
    """
    Filtre de tendance weekly (close > EMA) aligné sur l'index daily, sans
    lookahead : seule la dernière bougie weekly déjà clôturée est utilisée.
    """
    weekly_close = weekly_df["close"]
    weekly_ema = vbt.MA.run(weekly_close, window=ema_window, ewm=True).ma
    bullish = weekly_close > weekly_ema
    weekly_available_at = weekly_df.index + pd.Timedelta(days=7)

    weekly_signal_df = pd.DataFrame({"date": weekly_available_at, "bullish": bullish.values}).sort_values("date")
    weekly_signal_df["date"] = weekly_signal_df["date"].astype("datetime64[ns]")
    daily_df = pd.DataFrame({"date": daily_index}).sort_values("date")
    daily_df["date"] = daily_df["date"].astype("datetime64[ns]")
    merged = pd.merge_asof(daily_df, weekly_signal_df, on="date", direction="backward")
    merged["bullish"] = merged["bullish"].fillna(False).astype(bool)
    return pd.Series(merged["bullish"].values, index=pd.DatetimeIndex(merged["date"]))


@dataclass
class SignalState:
    asset: str
    date: pd.Timestamp
    close: float
    supertrend_direction: int
    weekly_bullish: bool
    ta_signal: str  # "BUY", "SELL", "HOLD"
    atr_stop_price: float | None  # None si pas de position ouverte (calculé par risk_management pour une position existante)


def latest_signal(asset: str, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> SignalState:
    """
    Calcule l'état du signal TA validé (Supertrend+Weekly) au dernier jour
    disponible. Utilisé par le moteur de décision de l'agent.
    """
    bull_cross, bear_cross = supertrend_crosses(daily_df)
    weekly_bullish = weekly_trend_filter(daily_df.index, weekly_df)

    long_entry = bull_cross & weekly_bullish
    long_exit = bear_cross | (~weekly_bullish)

    last_date = daily_df.index[-1]
    if bool(long_entry.iloc[-1]):
        ta_signal = "BUY"
    elif bool(long_exit.iloc[-1]):
        ta_signal = "SELL"
    else:
        ta_signal = "HOLD"

    return SignalState(
        asset=asset,
        date=last_date,
        close=float(daily_df["close"].iloc[-1]),
        supertrend_direction=int(supertrend(daily_df).iloc[-1]),
        weekly_bullish=bool(weekly_bullish.iloc[-1]),
        ta_signal=ta_signal,
        atr_stop_price=None,
    )
