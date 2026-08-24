"""
Étape 7 : Indicateurs supplémentaires les plus utilisés en trading crypto
Supertrend, ADX/DMI, Stochastic RSI, Parabolic SAR, Donchian Breakout, OBV,
CCI, Williams %R, EMA Golden Cross (50/200).
Paramètres classiques fixes (pas de grid-search) — testés directement sur la
même fenêtre hors-échantillon que l'étape 6 (walk-forward), pour rester
comparables sans répéter l'erreur de sur-optimisation in-sample.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
TRAIN_LEN = 540
TEST_LEN = 90
STEP = TEST_LEN

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


def oos_bounds(n, train_len=TRAIN_LEN, test_len=TEST_LEN, step=STEP):
    folds = []
    idx = 0
    while idx + train_len + test_len <= n:
        folds.append((idx, idx + train_len, idx + train_len + test_len))
        idx += step
    if not folds:
        return None
    return folds[0][1], folds[-1][2] - 1


# --- Indicateurs supplémentaires ------------------------------------------------

def signals_supertrend(df, atr_period=10, multiplier=3):
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

    direction_s = pd.Series(direction, index=close.index)
    entries = (direction_s == 1) & (direction_s.shift(1) == -1)
    exits = (direction_s == -1) & (direction_s.shift(1) == 1)
    return entries.fillna(False), exits.fillna(False)


def signals_adx_dmi(df, period=14, adx_threshold=25):
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    entries = plus_di.vbt.crossed_above(minus_di) & (adx > adx_threshold)
    exits = minus_di.vbt.crossed_above(plus_di)
    return entries, exits


def signals_stoch_rsi(close, rsi_period=14, stoch_period=14, lower=20, upper=80):
    rsi = vbt.RSI.run(close, window=rsi_period).rsi
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    k_line = stoch_rsi.rolling(3).mean()
    entries = k_line.vbt.crossed_above(lower)
    exits = k_line.vbt.crossed_below(upper)
    return entries, exits


def signals_parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(close)
    sar = np.zeros(n)
    trend = np.ones(n, dtype=int)
    af = af_start
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    trend_s = pd.Series(trend, index=df.index)
    entries = (trend_s == 1) & (trend_s.shift(1) == -1)
    exits = (trend_s == -1) & (trend_s.shift(1) == 1)
    return entries.fillna(False), exits.fillna(False)


def signals_donchian_breakout(df, window=20):
    high, low, close = df["high"], df["low"], df["close"]
    upper = high.rolling(window).max().shift(1)
    lower = low.rolling(window).min().shift(1)
    entries = close.vbt.crossed_above(upper)
    exits = close.vbt.crossed_below(lower)
    return entries, exits


def signals_obv_trend(df, ma_window=20):
    obv = vbt.OBV.run(df["close"], df["volume"]).obv
    obv_ma = obv.rolling(ma_window).mean()
    entries = obv.vbt.crossed_above(obv_ma)
    exits = obv.vbt.crossed_below(obv_ma)
    return entries, exits


def signals_cci(df, window=20, lower=-100, upper=100):
    high, low, close = df["high"], df["low"], df["close"]
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window).mean()
    mean_dev = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    entries = cci.vbt.crossed_above(upper)
    exits = cci.vbt.crossed_below(lower)
    return entries, exits


def signals_williams_r(df, window=14, lower=-80, upper=-20):
    high, low, close = df["high"], df["low"], df["close"]
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    wpr = -100 * (highest_high - close) / (highest_high - lowest_low)
    entries = wpr.vbt.crossed_above(lower)
    exits = wpr.vbt.crossed_below(upper)
    return entries, exits


def signals_ema_golden_cross(close, fast=50, slow=200):
    ema_fast = vbt.MA.run(close, window=fast, ewm=True).ma
    ema_slow = vbt.MA.run(close, window=slow, ewm=True).ma
    entries = ema_fast.vbt.crossed_above(ema_slow)
    exits = ema_fast.vbt.crossed_below(ema_slow)
    return entries, exits


INDICATORS_OHLCV = {
    "Supertrend_10_3": signals_supertrend,
    "ADX_DMI_14_25": signals_adx_dmi,
    "Parabolic_SAR": signals_parabolic_sar,
    "Donchian_Breakout_20": signals_donchian_breakout,
    "OBV_Trend_MA20": signals_obv_trend,
    "CCI_20": signals_cci,
    "Williams_R_14": signals_williams_r,
}

INDICATORS_CLOSE = {
    "StochRSI_14_14": signals_stoch_rsi,
    "EMA_Golden_Cross_50_200": signals_ema_golden_cross,
}


def score_portfolio(pf: vbt.Portfolio) -> dict:
    trades = pf.trades
    num_trades = trades.count()
    return {
        "rendement_net_pct": round(pf.total_return() * 100, 2),
        "profit_factor": round(trades.profit_factor(), 2) if num_trades > 0 else np.nan,
        "max_drawdown_pct": round(pf.max_drawdown() * 100, 2),
        "win_rate_pct": round(trades.win_rate() * 100, 2) if num_trades > 0 else np.nan,
        "nb_trades": num_trades,
    }


if __name__ == "__main__":
    results = []

    for asset, csv_file in ASSETS.items():
        df = load_ohlcv(csv_file)
        close = df["close"]
        bounds = oos_bounds(len(df))
        if bounds is None:
            print(f"\n--- {asset} : historique insuffisant ---")
            continue
        start_pos, end_pos = bounds
        oos_start, oos_end = close.index[start_pos], close.index[end_pos]
        print(f"\n--- {asset} : période hors-échantillon {oos_start.date()} -> {oos_end.date()} ---")

        for name, fn in INDICATORS_OHLCV.items():
            entries, exits = fn(df)
            entries_oos = entries.loc[oos_start:oos_end]
            exits_oos = exits.loc[oos_start:oos_end]
            close_oos = close.loc[oos_start:oos_end]
            pf = vbt.Portfolio.from_signals(close_oos, entries_oos, exits_oos, init_cash=INIT_CASH, fees=FEES, freq="1D")
            row = {"asset": asset, "indicator": name}
            row.update(score_portfolio(pf))
            results.append(row)

        for name, fn in INDICATORS_CLOSE.items():
            entries, exits = fn(close)
            entries_oos = entries.loc[oos_start:oos_end]
            exits_oos = exits.loc[oos_start:oos_end]
            close_oos = close.loc[oos_start:oos_end]
            pf = vbt.Portfolio.from_signals(close_oos, entries_oos, exits_oos, init_cash=INIT_CASH, fees=FEES, freq="1D")
            row = {"asset": asset, "indicator": name}
            row.update(score_portfolio(pf))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("extended_indicators_walkforward_results.csv", index=False)

    print("\n=== Résultats hors-échantillon par actif / indicateur ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par indicateur (tous actifs confondus) ===")
    avg = (
        results_df.groupby("indicator")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
