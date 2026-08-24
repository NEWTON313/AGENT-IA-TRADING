"""
Étape 15 : Weekly(EMA10)+Daily Supertrend en version Long/Short symétrique
Miroir exact de la version long-only (étape 10/15) :
  - Long  : Supertrend daily bascule haussier ET tendance weekly haussière
  - Short : Supertrend daily bascule baissier ET tendance weekly baissière
  - Sortie long  : Supertrend daily bascule baissier OU weekly repasse baissier
  - Sortie short : Supertrend daily bascule haussier OU weekly repasse haussier
Comparé à la version long-only sur la même période hors-échantillon.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006

OOS_BOUNDS = {
    "BTC/USD": ("2023-02-16", "2026-07-29"),
    "ETH/USD": ("2023-02-16", "2026-07-29"),
    "SOL/USD": ("2023-02-16", "2026-07-29"),
    "XRP/USD": ("2025-01-03", "2026-06-26"),
}

ASSETS = {
    "BTC/USD": ("btc_usd_daily_full.csv", "btc_usd_weekly_full.csv"),
    "ETH/USD": ("eth_usd_daily_full.csv", "eth_usd_weekly_full.csv"),
    "SOL/USD": ("sol_usd_daily_full.csv", "sol_usd_weekly_full.csv"),
    "XRP/USD": ("xrp_usd_daily_full.csv", "xrp_usd_weekly_full.csv"),
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


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
    bull_cross = (direction_s == 1) & (direction_s.shift(1) == -1)
    bear_cross = (direction_s == -1) & (direction_s.shift(1) == 1)
    return bull_cross.fillna(False), bear_cross.fillna(False)


def weekly_state_on_daily(daily_index, weekly_df, ema_window=10):
    weekly_close = weekly_df["close"]
    weekly_ema = vbt.MA.run(weekly_close, window=ema_window, ewm=True).ma
    bullish = weekly_close > weekly_ema
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
        oos_start, oos_end = OOS_BOUNDS[asset]

        st_bull_cross, st_bear_cross = signals_supertrend(daily_df)
        weekly_bullish = weekly_state_on_daily(daily_df.index, weekly_df, ema_window=10)

        long_entries = st_bull_cross & weekly_bullish
        long_exits = st_bear_cross | (~weekly_bullish)
        short_entries = st_bear_cross & (~weekly_bullish)
        short_exits = st_bull_cross | weekly_bullish

        close_oos = close.loc[oos_start:oos_end]

        pf_long_only = vbt.Portfolio.from_signals(
            close_oos,
            entries=long_entries.loc[oos_start:oos_end],
            exits=long_exits.loc[oos_start:oos_end],
            init_cash=INIT_CASH, fees=FEES, freq="1D",
        )

        pf_long_short = vbt.Portfolio.from_signals(
            close_oos,
            entries=long_entries.loc[oos_start:oos_end],
            exits=long_exits.loc[oos_start:oos_end],
            short_entries=short_entries.loc[oos_start:oos_end],
            short_exits=short_exits.loc[oos_start:oos_end],
            init_cash=INIT_CASH, fees=FEES, freq="1D",
        )

        row_long = {"asset": asset, "variante": "Long_only"}
        row_long.update(score_portfolio(pf_long_only))
        results.append(row_long)

        row_ls = {"asset": asset, "variante": "Long_Short_symetrique"}
        row_ls.update(score_portfolio(pf_long_short))
        results.append(row_ls)

        print(f"\n--- {asset} ---")
        print(f"  Long_only        : {row_long}")
        print(f"  Long_Short       : {row_ls}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("long_short_comparison_results.csv", index=False)

    print("\n=== Long-only vs Long/Short — résultats par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par variante ===")
    avg = (
        results_df.groupby("variante")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
