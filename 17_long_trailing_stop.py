"""
Étape 16 : Weekly(EMA10)+Daily Supertrend (long-only) + stop suiveur ATR
La version validée (étape 10/15) sort uniquement sur signal (Supertrend
baissier ou tendance weekly retournée). On teste l'ajout d'un stop suiveur
ATR (Chandelier Exit) en plus du signal — la sortie se déclenche sur le
premier des deux événements, pour verrouiller les gains plus tôt en cas de
retournement brutal avant que le signal ne se déclenche.
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

ATR_MULTIPLIERS = [2.0, 3.0, 4.0]


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


def weekly_bullish_on_daily(daily_index, weekly_df, ema_window=10):
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
        weekly_bullish = weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=10)

        entries = (st_bull_cross & weekly_bullish).loc[oos_start:oos_end]
        exits_signal = (st_bear_cross | (~weekly_bullish)).loc[oos_start:oos_end]
        close_oos = close.loc[oos_start:oos_end]

        atr14 = vbt.ATR.run(daily_df["high"], daily_df["low"], daily_df["close"], window=14).atr

        pf_baseline = vbt.Portfolio.from_signals(
            close_oos, entries=entries, exits=exits_signal,
            init_cash=INIT_CASH, fees=FEES, freq="1D",
        )
        row = {"asset": asset, "variante": "Signal_seul (etape 10)"}
        row.update(score_portfolio(pf_baseline))
        results.append(row)

        for mult in ATR_MULTIPLIERS:
            atr_pct = (mult * atr14 / close).fillna(0.10).loc[oos_start:oos_end]
            pf_trail = vbt.Portfolio.from_signals(
                close_oos, entries=entries, exits=exits_signal,
                sl_stop=atr_pct, sl_trail=True,
                init_cash=INIT_CASH, fees=FEES, freq="1D",
            )
            row = {"asset": asset, "variante": f"Signal+Trailing_ATR_x{mult}"}
            row.update(score_portfolio(pf_trail))
            results.append(row)

        print(f"\n--- {asset} terminé ---")

    results_df = pd.DataFrame(results)
    results_df.to_csv("long_trailing_stop_comparison.csv", index=False)

    print("\n=== Signal seul vs Signal+Trailing ATR — résultats par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par variante ===")
    avg = (
        results_df.groupby("variante")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
