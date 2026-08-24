"""
Étape 5 : Stratégies combinées (filtre de tendance + confirmation)
Combine des indicateurs pour réduire le bruit : un filtre de tendance (SMA
longue ou Ichimoku) couplé à un déclencheur de momentum (RSI) ou de
retournement (Bollinger + MACD). Paramètres volontairement standards
(non sur-optimisés) pour éviter d'empiler du surajustement sur le
grid-search de l'étape 4.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
MIN_TRADES = 5

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


def ichimoku_lines(close, tenkan_w=9, kijun_w=26, senkou_w=52):
    tenkan = (close.rolling(tenkan_w).max() + close.rolling(tenkan_w).min()) / 2
    kijun = (close.rolling(kijun_w).max() + close.rolling(kijun_w).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_w)
    senkou_b = ((close.rolling(senkou_w).max() + close.rolling(senkou_w).min()) / 2).shift(kijun_w)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    return tenkan, kijun, cloud_top


def strategy_trend_filtered_rsi(close, sma_trend_w=150, rsi_w=14, lower=30, upper=70):
    sma_trend = vbt.MA.run(close, window=sma_trend_w).ma
    rsi = vbt.RSI.run(close, window=rsi_w).rsi
    uptrend = close > sma_trend
    entries = uptrend & rsi.vbt.crossed_above(lower)
    exits = rsi.vbt.crossed_below(upper) | close.vbt.crossed_below(sma_trend)
    return entries, exits


def strategy_ichimoku_filtered_rsi(close, rsi_w=14, lower=30, upper=70):
    _, _, cloud_top = ichimoku_lines(close)
    rsi = vbt.RSI.run(close, window=rsi_w).rsi
    uptrend = close > cloud_top
    entries = uptrend & rsi.vbt.crossed_above(lower)
    exits = rsi.vbt.crossed_below(upper) | ~uptrend
    return entries, exits


def strategy_macd_bbands_reversion(close, fast=12, slow=26, signal=9, bb_window=20, alpha=2.0):
    macd_ind = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
    bullish_macd = macd_ind.macd > macd_ind.signal
    bb = vbt.BBANDS.run(close, window=bb_window, alpha=alpha)
    entries = bullish_macd & close.vbt.crossed_below(bb.lower)
    exits = close.vbt.crossed_above(bb.middle) | ~bullish_macd
    return entries, exits


STRATEGIES = {
    "Trend(SMA150)+RSI(14,30,70)": strategy_trend_filtered_rsi,
    "Ichimoku(9,26,52)+RSI(14,30,70)": strategy_ichimoku_filtered_rsi,
    "MACD(12,26,9)+Bollinger(20,2)": strategy_macd_bbands_reversion,
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

        for strat_name, strat_fn in STRATEGIES.items():
            entries, exits = strat_fn(close)
            pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=INIT_CASH, fees=FEES, freq="1D")
            row = {"asset": asset, "strategy": strat_name}
            row.update(score_portfolio(pf))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("combined_strategies_scores.csv", index=False)

    print("=== Stratégies combinées — scores par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par stratégie (tous actifs confondus) ===")
    avg = (
        results_df.groupby("strategy")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())

    print(f"\nRappel — meilleur indicateur isolé (étape 4, in-sample) : RSI_Reversal(14) en tête sur les 4 actifs.")
    print("Comparer les deux tableaux (combined_strategies_scores.csv vs indicator_optimization_best.csv).")
