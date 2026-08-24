"""
Étape 14 : Comportement de Weekly(EMA10)+Daily Supertrend en marché baissier
Identifie les phases de marché baissier au sein de la période hors-échantillon
(drawdown >= 20% depuis le plus haut glissant sur 180 jours, épisodes d'au
moins 30 jours), puis compare la performance de la stratégie sur ces phases
précises à un simple Buy & Hold. Objectif : vérifier si la stratégie protège
réellement le capital en tendance baissière (elle est long-only, donc censée
rester en cash), plutôt que de le supposer.
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
    entries = (direction_s == 1) & (direction_s.shift(1) == -1)
    exits = (direction_s == -1) & (direction_s.shift(1) == 1)
    return entries.fillna(False), exits.fillna(False)


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


def find_bear_episodes(close: pd.Series, dd_threshold=0.20, min_days=30, peak_window=180):
    rolling_peak = close.rolling(peak_window, min_periods=1).max()
    drawdown = (close - rolling_peak) / rolling_peak
    in_bear = drawdown <= -dd_threshold

    episodes = []
    start = None
    for date, flag in in_bear.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            if (date - start).days >= min_days:
                episodes.append((start, date))
            start = None
    if start is not None and (in_bear.index[-1] - start).days >= min_days:
        episodes.append((start, in_bear.index[-1]))
    return episodes


if __name__ == "__main__":
    all_episode_rows = []

    for asset, (daily_csv, weekly_csv) in ASSETS.items():
        daily_df = load_ohlcv(daily_csv)
        weekly_df = load_ohlcv(weekly_csv)
        close = daily_df["close"]
        oos_start, oos_end = OOS_BOUNDS[asset]
        close_oos_full = close.loc[oos_start:oos_end]

        st_entries, st_exits = signals_supertrend(daily_df)
        weekly_bullish = weekly_bullish_on_daily(daily_df.index, weekly_df, ema_window=10)
        entries = (st_entries & weekly_bullish).loc[oos_start:oos_end]
        exits = (st_exits | (~weekly_bullish)).loc[oos_start:oos_end]

        pf = vbt.Portfolio.from_signals(close_oos_full, entries, exits, init_cash=INIT_CASH, fees=FEES, freq="1D")
        strategy_equity = pf.value()

        bh_pf = vbt.Portfolio.from_holding(close_oos_full, init_cash=INIT_CASH, fees=FEES, freq="1D")
        bh_equity = bh_pf.value()

        episodes = find_bear_episodes(close_oos_full)
        print(f"\n--- {asset} : {len(episodes)} épisode(s) de marché baissier détecté(s) (drawdown >= 20%, >= 30j) ---")

        if not episodes:
            print("  Aucun épisode de baisse soutenue détecté sur cette période.")
            continue

        for ep_start, ep_end in episodes:
            bh_ret = (bh_equity.loc[ep_end] / bh_equity.loc[ep_start] - 1) * 100
            strat_ret = (strategy_equity.loc[ep_end] / strategy_equity.loc[ep_start] - 1) * 100
            price_dd = (close_oos_full.loc[ep_start:ep_end].min() / close_oos_full.loc[ep_start] - 1) * 100
            row = {
                "asset": asset,
                "episode_debut": ep_start.date(),
                "episode_fin": ep_end.date(),
                "duree_jours": (ep_end - ep_start).days,
                "chute_prix_max_pct": round(price_dd, 2),
                "buy_and_hold_pct": round(bh_ret, 2),
                "strategie_pct": round(strat_ret, 2),
                "strategie_protege": strat_ret > bh_ret,
            }
            all_episode_rows.append(row)
            print(f"  {ep_start.date()} -> {ep_end.date()} ({row['duree_jours']}j) | "
                  f"chute prix: {row['chute_prix_max_pct']}% | "
                  f"Buy&Hold: {row['buy_and_hold_pct']}% | "
                  f"Stratégie: {row['strategie_pct']}% | "
                  f"{'✅ protège' if row['strategie_protege'] else '❌ ne protège pas'}")

    if all_episode_rows:
        episodes_df = pd.DataFrame(all_episode_rows)
        episodes_df.to_csv("bear_market_episodes_analysis.csv", index=False)
        print("\n=== Synthèse de tous les épisodes baissiers ===")
        print(episodes_df.to_string(index=False))
        print(f"\nStratégie protège dans {episodes_df['strategie_protege'].sum()}/{len(episodes_df)} épisodes.")
        print(f"Moyenne Buy&Hold sur épisodes baissiers : {episodes_df['buy_and_hold_pct'].mean():.2f}%")
        print(f"Moyenne Stratégie sur épisodes baissiers : {episodes_df['strategie_pct'].mean():.2f}%")
