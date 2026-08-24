"""
Étape 3 : Backtesting & scoring des indicateurs (isolés)
Teste chaque indicateur technique séparément (SMA cross, MACD, RSI, Bollinger
Bands, Ichimoku) sur chaque actif, en position long-only 100% cash in/out,
avec des frais Coinbase Advanced standard.
Métriques de scoring : Rendement net, Profit Factor, Max Drawdown, Win Rate.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006  # 0.6% taker, hypothèse Coinbase Advanced standard

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


# --- Génération des signaux entrée/sortie, un indicateur à la fois -------------

def signals_sma_cross(close: pd.Series, fast: int = 20, slow: int = 50):
    sma_fast = vbt.MA.run(close, window=fast).ma
    sma_slow = vbt.MA.run(close, window=slow).ma
    entries = sma_fast.vbt.crossed_above(sma_slow)
    exits = sma_fast.vbt.crossed_below(sma_slow)
    return entries, exits


def signals_macd_cross(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_ind = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
    entries = macd_ind.macd.vbt.crossed_above(macd_ind.signal)
    exits = macd_ind.macd.vbt.crossed_below(macd_ind.signal)
    return entries, exits


def signals_rsi_reversal(close: pd.Series, window: int = 14, lower: int = 30, upper: int = 70):
    rsi = vbt.RSI.run(close, window=window).rsi
    entries = rsi.vbt.crossed_above(lower)
    exits = rsi.vbt.crossed_below(upper)
    return entries, exits


def signals_bbands_reversion(close: pd.Series, window: int = 20, alpha: int = 2):
    bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
    entries = close.vbt.crossed_below(bb.lower)
    exits = close.vbt.crossed_above(bb.middle)
    return entries, exits


def signals_ichimoku(close: pd.Series, tenkan_w: int = 9, kijun_w: int = 26, senkou_w: int = 52):
    high = close.rolling(tenkan_w).max()
    low = close.rolling(tenkan_w).min()
    tenkan = (high + low) / 2

    high_k = close.rolling(kijun_w).max()
    low_k = close.rolling(kijun_w).min()
    kijun = (high_k + low_k) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(kijun_w)
    high_s = close.rolling(senkou_w).max()
    low_s = close.rolling(senkou_w).min()
    senkou_b = ((high_s + low_s) / 2).shift(kijun_w)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    above_cloud = close > cloud_top
    tenkan_cross_up = tenkan.vbt.crossed_above(kijun)
    tenkan_cross_down = tenkan.vbt.crossed_below(kijun)

    entries = tenkan_cross_up & above_cloud
    exits = tenkan_cross_down | (close < cloud_bottom)
    return entries, exits


INDICATORS = {
    "SMA_Cross_20_50": signals_sma_cross,
    "MACD_12_26_9": signals_macd_cross,
    "RSI_14_Reversal": signals_rsi_reversal,
    "Bollinger_20_2_Reversion": signals_bbands_reversion,
    "Ichimoku_9_26_52": signals_ichimoku,
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

        for indicator_name, signal_fn in INDICATORS.items():
            entries, exits = signal_fn(close)
            pf = vbt.Portfolio.from_signals(
                close,
                entries,
                exits,
                init_cash=INIT_CASH,
                fees=FEES,
                freq="1D",
            )
            row = {"asset": asset, "indicator": indicator_name}
            row.update(score_portfolio(pf))
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv("indicator_backtest_scores.csv", index=False)

    print("\n=== Scores par actif / indicateur ===")
    print(results_df.to_string(index=False))

    print("\n=== Meilleur indicateur par actif (rendement net) ===")
    best_per_asset = results_df.loc[results_df.groupby("asset")["rendement_net_pct"].idxmax()]
    print(best_per_asset.to_string(index=False))

    print("\n=== Classement moyen par indicateur (tous actifs confondus) ===")
    avg_by_indicator = (
        results_df.groupby("indicator")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg_by_indicator.to_string())
