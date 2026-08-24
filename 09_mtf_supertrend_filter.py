"""
Étape 9 : Filtre multi-timeframe — tendance Weekly + Supertrend Daily
Le signal de trading reste sur Daily (Supertrend 10/3, meilleur candidat de
l'étape 7), mais n'est autorisé que dans le sens de la tendance Weekly
(close > EMA20 weekly). Alignement sans biais de lookahead : seule la
dernière bougie hebdo déjà CLÔTURÉE à la date daily courante est utilisée.
Comparé au Supertrend daily seul, sur la même période hors-échantillon
que les étapes 6/7.
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


def weekly_trend_filter_on_daily(daily_index: pd.DatetimeIndex, weekly_df: pd.DataFrame, ema_window: int = 20) -> pd.Series:
    """
    Filtre de tendance weekly (close > EMA20) aligné sur l'index daily,
    sans lookahead : seule la dernière bougie weekly déjà close est utilisée.
    """
    weekly_close = weekly_df["close"]
    weekly_ema = vbt.MA.run(weekly_close, window=ema_window, ewm=True).ma
    weekly_bullish = weekly_close > weekly_ema
    weekly_available_at = weekly_df.index + pd.Timedelta(days=7)  # date à laquelle la bougie weekly est connue

    weekly_signal_df = pd.DataFrame({"date": weekly_available_at, "bullish": weekly_bullish.values}).sort_values("date")
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
            print(f"\n--- {asset} : historique insuffisant ---")
            continue
        start_pos, end_pos = bounds
        oos_start, oos_end = close.index[start_pos], close.index[end_pos]
        print(f"\n--- {asset} : période hors-échantillon {oos_start.date()} -> {oos_end.date()} ---")

        st_entries, st_exits = signals_supertrend(daily_df)
        weekly_bullish = weekly_trend_filter_on_daily(daily_df.index, weekly_df, ema_window=20)

        mtf_entries = st_entries & weekly_bullish
        mtf_exits = st_exits | (~weekly_bullish)

        variants = {
            "Supertrend_daily_seul": (st_entries, st_exits),
            "Supertrend_daily+Weekly_trend_filter": (mtf_entries, mtf_exits),
        }

        for name, (entries, exits) in variants.items():
            entries_oos = entries.loc[oos_start:oos_end]
            exits_oos = exits.loc[oos_start:oos_end]
            close_oos = close.loc[oos_start:oos_end]
            pf = vbt.Portfolio.from_signals(close_oos, entries_oos, exits_oos, init_cash=INIT_CASH, fees=FEES, freq="1D")
            row = {"asset": asset, "variant": name}
            row.update(score_portfolio(pf))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("mtf_supertrend_results.csv", index=False)

    print("\n=== Supertrend daily seul vs. filtré par tendance Weekly ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par variante ===")
    avg = (
        results_df.groupby("variant")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
