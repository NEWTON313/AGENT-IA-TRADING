"""
Étape 10 : Assouplissement du filtre de tendance Weekly
Le filtre EMA20 weekly (étape 9) réduit le drawdown mais coûte beaucoup de
rendement (retard sur les retournements). On teste des variantes moins
agressives : EMA plus courte (réaction plus rapide), et une version qui
autorise l'entrée si la tendance weekly vient tout juste de basculer haussière
(capte le début du mouvement au lieu d'attendre sa confirmation complète).
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
    "BTC/USD": ("btc_usd_daily_full.csv", "btc_usd_weekly_full.csv"),
    "ETH/USD": ("eth_usd_daily_full.csv", "eth_usd_weekly_full.csv"),
    "SOL/USD": ("sol_usd_daily_full.csv", "sol_usd_weekly_full.csv"),
    "XRP/USD": ("xrp_usd_daily_full.csv", "xrp_usd_weekly_full.csv"),
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


def weekly_bullish_on_daily(daily_index, weekly_df, ema_window, recent_turn_weeks=None):
    """
    Filtre weekly aligné sur l'index daily (sans lookahead).
    Si recent_turn_weeks est fourni : reste bullish si la tendance a basculé
    haussière il y a <= recent_turn_weeks semaines (capte le début du move).
    """
    weekly_close = weekly_df["close"]
    weekly_ema = vbt.MA.run(weekly_close, window=ema_window, ewm=True).ma
    bullish = (weekly_close > weekly_ema)

    if recent_turn_weeks is not None:
        just_turned = bullish & (~bullish.shift(1).fillna(False))
        recent_flag = just_turned.rolling(recent_turn_weeks, min_periods=1).max().astype(bool)
        bullish = bullish | recent_flag

    weekly_available_at = weekly_df.index + pd.Timedelta(days=7)
    weekly_signal_df = pd.DataFrame({"date": weekly_available_at, "bullish": bullish.values}).sort_values("date")
    daily_df = pd.DataFrame({"date": daily_index}).sort_values("date")
    merged = pd.merge_asof(daily_df, weekly_signal_df, on="date", direction="backward")
    merged["bullish"] = merged["bullish"].fillna(False).astype(bool)
    return pd.Series(merged["bullish"].values, index=pd.DatetimeIndex(merged["date"]))


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

    for asset, (daily_csv, weekly_csv) in ASSETS.items():
        daily_df = load_ohlcv(daily_csv)
        weekly_df = load_ohlcv(weekly_csv)
        close = daily_df["close"]

        bounds = oos_bounds(len(daily_df))
        if bounds is None:
            continue
        start_pos, end_pos = bounds
        oos_start, oos_end = close.index[start_pos], close.index[end_pos]
        print(f"\n--- {asset} : période hors-échantillon {oos_start.date()} -> {oos_end.date()} ---")

        st_entries, st_exits = signals_supertrend(daily_df)

        filters = {
            "Sans_filtre": pd.Series(True, index=close.index),
            "Weekly_EMA20 (etape 9)": weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=20),
            "Weekly_EMA10": weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=10),
            "Weekly_EMA20+retournement_recent(2sem)": weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=20, recent_turn_weeks=2),
        }

        for filt_name, weekly_bullish in filters.items():
            entries = st_entries & weekly_bullish
            exits = st_exits | (~weekly_bullish)
            entries_oos = entries.loc[oos_start:oos_end]
            exits_oos = exits.loc[oos_start:oos_end]
            close_oos = close.loc[oos_start:oos_end]
            pf = vbt.Portfolio.from_signals(close_oos, entries_oos, exits_oos, init_cash=INIT_CASH, fees=FEES, freq="1D")
            row = {"asset": asset, "filtre": filt_name}
            row.update(score_portfolio(pf))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("weekly_filter_variants_results.csv", index=False)

    print("\n=== Variantes de filtre weekly — résultats par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par variante ===")
    avg = (
        results_df.groupby("filtre")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
