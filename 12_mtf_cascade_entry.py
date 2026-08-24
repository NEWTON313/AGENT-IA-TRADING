"""
Étape 11b : Cascade multi-timeframe — Weekly (macro) > Daily (tendance) > 4h (timing)
- Weekly EMA10 bullish (meilleur filtre trouvé à l'étape 10) = feu vert macro
- Daily Supertrend(10,3) bullish = confirmation de tendance intermédiaire
- 4h Supertrend(10,3) qui bascule haussier = déclencheur d'entrée précis
Sortie dès que l'un des trois niveaux repasse baissier. Objectif : récupérer
une partie du rendement perdu par le lag du filtre weekly (étape 9/10) grâce
à un timing d'entrée plus fin, sans perdre la protection sur le drawdown.
Exécution au niveau 4h, comparée à la même période hors-échantillon que les
étapes 6/7/9/10.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006

# Bornes hors-échantillon identiques aux étapes 6/7/9/10
OOS_BOUNDS = {
    "BTC/USD": ("2023-02-16", "2026-07-29"),
    "ETH/USD": ("2023-02-16", "2026-07-29"),
    "SOL/USD": ("2023-02-16", "2026-07-29"),
    "XRP/USD": ("2025-01-03", "2026-06-26"),
}

ASSETS = {
    "BTC/USD": ("btc_usd_daily_full.csv", "btc_usd_weekly_full.csv", "btc_usd_4h_oos.csv"),
    "ETH/USD": ("eth_usd_daily_full.csv", "eth_usd_weekly_full.csv", "eth_usd_4h_oos.csv"),
    "SOL/USD": ("sol_usd_daily_full.csv", "sol_usd_weekly_full.csv", "sol_usd_4h_oos.csv"),
    "XRP/USD": ("xrp_usd_daily_full.csv", "xrp_usd_weekly_full.csv", "xrp_usd_4h_oos.csv"),
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
    entries = (direction_s == 1) & (direction_s.shift(1) == -1)
    exits = (direction_s == -1) & (direction_s.shift(1) == 1)
    return direction_s, entries.fillna(False), exits.fillna(False)


def align_backward_no_lookahead(target_index, source_index, source_values, available_delay):
    """
    Aligne une série (daily ou weekly) sur un index plus fin (4h), sans
    lookahead : la valeur d'une bougie source n'est utilisable qu'à partir
    de available_delay après son timestamp (le temps qu'elle soit clôturée).
    """
    available_at = source_index + available_delay
    source_df = pd.DataFrame({"date": available_at, "value": source_values}).sort_values("date")
    target_df = pd.DataFrame({"date": target_index}).sort_values("date")
    merged = pd.merge_asof(target_df, source_df, on="date", direction="backward")
    merged["value"] = merged["value"].fillna(False).astype(bool)
    return pd.Series(merged["value"].values, index=pd.DatetimeIndex(merged["date"]))


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

    for asset, (daily_csv, weekly_csv, h4_csv) in ASSETS.items():
        daily_df = load_ohlcv(daily_csv)
        weekly_df = load_ohlcv(weekly_csv)
        h4_df = load_ohlcv(h4_csv)
        oos_start, oos_end = OOS_BOUNDS[asset]

        # Filtre weekly EMA10 (meilleur trouvé à l'étape 10)
        weekly_ema10 = vbt.MA.run(weekly_df["close"], window=10, ewm=True).ma
        weekly_bullish = weekly_df["close"] > weekly_ema10

        # Tendance daily Supertrend
        daily_direction, _, _ = signals_supertrend(daily_df)
        daily_bullish = daily_direction == 1

        # Timing d'entrée/sortie 4h Supertrend
        h4_direction, h4_entries, h4_exits = signals_supertrend(h4_df)

        weekly_bullish_4h = align_backward_no_lookahead(h4_df.index, weekly_df.index, weekly_bullish.values, pd.Timedelta(days=7))
        daily_bullish_4h = align_backward_no_lookahead(h4_df.index, daily_df.index, daily_bullish.values, pd.Timedelta(days=1))

        cascade_entries = h4_entries & daily_bullish_4h & weekly_bullish_4h
        cascade_exits = h4_exits | (~daily_bullish_4h) | (~weekly_bullish_4h)

        close_4h = h4_df["close"]
        entries_oos = cascade_entries.loc[oos_start:oos_end]
        exits_oos = cascade_exits.loc[oos_start:oos_end]
        close_oos = close_4h.loc[oos_start:oos_end]

        pf = vbt.Portfolio.from_signals(close_oos, entries_oos, exits_oos, init_cash=INIT_CASH, fees=FEES, freq="4h")
        row = {"asset": asset, "strategie": "Cascade_Weekly(EMA10)>Daily(ST)>4h(ST_timing)"}
        row.update(score_portfolio(pf))
        results.append(row)
        print(f"\n--- {asset} : {row}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("mtf_cascade_4h_results.csv", index=False)

    print("\n=== Cascade Weekly>Daily>4h — résultats par actif ===")
    print(results_df.to_string(index=False))
    print("\nRappel comparatif (étape 10, exécution daily) :")
    print("  Sans_filtre        : rendement moyen +78.71% | maxDD -50.21% | WR 30.2% | trades~13")
    print("  Weekly_EMA10 (daily): rendement moyen +23.92% | maxDD -28.88% | WR 56.7% | trades~3.5")
