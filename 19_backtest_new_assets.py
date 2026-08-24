"""
Étape 19 : Validation de la stratégie Weekly(EMA10)+Daily Supertrend(10,3)+
Trailing Stop ATR(14)x3 sur les 9 nouveaux actifs candidats. Paramètres
fixes (aucun grid-search) — pas besoin de split walk-forward train/test,
juste un backtest complet sur tout l'historique disponible de chaque actif.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
ATR_STOP_MULT = 3.0

ASSETS = {
    "ADA/USD": "ada_usd",
    "DOGE/USD": "doge_usd",
    "AAVE/USD": "aave_usd",
    "DOT/USD": "dot_usd",
    "LINK/USD": "link_usd",
    "TRUMP/USD": "trump_usd",
    "FET/USD": "fet_usd",
    "POL/USD": "pol_usd",
    "FLOW/USD": "flow_usd",
}


def load_ohlcv(stub: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(f"{stub}_{kind}_full.csv", index_col="timestamp", parse_dates=True)


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


def weekly_bullish_on_daily(daily_index, weekly_df, ema_window=10):
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

    for asset, stub in ASSETS.items():
        daily_df = load_ohlcv(stub, "daily")
        weekly_df = load_ohlcv(stub, "weekly")
        close = daily_df["close"]

        st_entries, st_exits = signals_supertrend(daily_df)
        weekly_bullish = weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=10)

        entries = st_entries & weekly_bullish
        exits_signal = st_exits | (~weekly_bullish)

        atr14 = vbt.ATR.run(daily_df["high"], daily_df["low"], daily_df["close"], window=14).atr
        atr_pct = (ATR_STOP_MULT * atr14 / close).fillna(0.10)

        pf = vbt.Portfolio.from_signals(
            close, entries=entries, exits=exits_signal,
            sl_stop=atr_pct, sl_trail=True,
            init_cash=INIT_CASH, fees=FEES, freq="1D",
        )
        row = {"asset": asset, "historique_jours": len(daily_df)}
        row.update(score_portfolio(pf))
        results.append(row)
        print(f"{asset:10s} ({len(daily_df)}j) -> {row}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("new_assets_validation_results.csv", index=False)

    print("\n=== Résultats complets ===")
    print(results_df.to_string(index=False))

    print("\nRappel — actifs déjà validés (étape 10, walk-forward) :")
    print("  Moyenne BTC/ETH/SOL/XRP : rendement +23.92% | WR 56.7% | PF 2.02 | maxDD -28.88%")
