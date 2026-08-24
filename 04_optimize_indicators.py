"""
Étape 4 : Optimisation des paramètres par indicateur (grid-search)
Balaie une grille de paramètres pour chaque indicateur, sur chaque actif,
et retient la meilleure combinaison (rendement net, sous contrainte d'un
nombre minimum de trades pour éviter le surajustement sur peu d'échantillons).
"""

import itertools
import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006  # 0.6% taker, hypothèse Coinbase Advanced standard
MIN_TRADES = 10  # sous ce seuil, le résultat est jugé non significatif et écarté

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


# --- Grilles de paramètres et générateurs de signaux --------------------------

def grid_sma_cross():
    for fast, slow in itertools.product([10, 20, 30], [50, 100, 150]):
        if fast < slow:
            yield {"fast": fast, "slow": slow}


def signals_sma_cross(close, fast, slow):
    sma_fast = vbt.MA.run(close, window=fast).ma
    sma_slow = vbt.MA.run(close, window=slow).ma
    return sma_fast.vbt.crossed_above(sma_slow), sma_fast.vbt.crossed_below(sma_slow)


def grid_macd():
    for fast, slow, signal in itertools.product([8, 12, 16], [21, 26, 30], [5, 9, 12]):
        if fast < slow:
            yield {"fast": fast, "slow": slow, "signal": signal}


def signals_macd(close, fast, slow, signal):
    macd_ind = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
    return macd_ind.macd.vbt.crossed_above(macd_ind.signal), macd_ind.macd.vbt.crossed_below(macd_ind.signal)


def grid_rsi():
    for window, lower, upper in itertools.product([7, 10, 14, 21], [20, 25, 30, 35], [65, 70, 75, 80]):
        yield {"window": window, "lower": lower, "upper": upper}


def signals_rsi(close, window, lower, upper):
    rsi = vbt.RSI.run(close, window=window).rsi
    return rsi.vbt.crossed_above(lower), rsi.vbt.crossed_below(upper)


def grid_bbands():
    for window, alpha in itertools.product([10, 20, 30], [1.5, 2.0, 2.5]):
        yield {"window": window, "alpha": alpha}


def signals_bbands(close, window, alpha):
    bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
    return close.vbt.crossed_below(bb.lower), close.vbt.crossed_above(bb.middle)


def grid_ichimoku():
    for tenkan, kijun, senkou in [(7, 22, 44), (9, 26, 52), (12, 30, 60)]:
        yield {"tenkan": tenkan, "kijun": kijun, "senkou": senkou}


def signals_ichimoku(close, tenkan, kijun, senkou):
    high = close.rolling(tenkan).max()
    low = close.rolling(tenkan).min()
    tenkan_line = (high + low) / 2

    high_k = close.rolling(kijun).max()
    low_k = close.rolling(kijun).min()
    kijun_line = (high_k + low_k) / 2

    senkou_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    high_s = close.rolling(senkou).max()
    low_s = close.rolling(senkou).min()
    senkou_b = ((high_s + low_s) / 2).shift(kijun)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    above_cloud = close > cloud_top
    entries = tenkan_line.vbt.crossed_above(kijun_line) & above_cloud
    exits = tenkan_line.vbt.crossed_below(kijun_line) | (close < cloud_bottom)
    return entries, exits


INDICATORS = {
    "SMA_Cross": (grid_sma_cross, signals_sma_cross),
    "MACD": (grid_macd, signals_macd),
    "RSI_Reversal": (grid_rsi, signals_rsi),
    "Bollinger_Reversion": (grid_bbands, signals_bbands),
    "Ichimoku": (grid_ichimoku, signals_ichimoku),
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
    all_results = []

    for asset, csv_file in ASSETS.items():
        df = load_ohlcv(csv_file)
        close = df["close"]
        print(f"\n--- Optimisation sur {asset} ---")

        for indicator_name, (grid_fn, signal_fn) in INDICATORS.items():
            best_row = None
            for params in grid_fn():
                entries, exits = signal_fn(close, **params)
                pf = vbt.Portfolio.from_signals(
                    close, entries, exits, init_cash=INIT_CASH, fees=FEES, freq="1D"
                )
                row = {"asset": asset, "indicator": indicator_name, "params": params}
                row.update(score_portfolio(pf))
                all_results.append(row)

                if row["nb_trades"] >= MIN_TRADES:
                    if best_row is None or row["rendement_net_pct"] > best_row["rendement_net_pct"]:
                        best_row = row

            if best_row:
                print(f"  {indicator_name:22s} -> {best_row['params']} | "
                      f"rendement={best_row['rendement_net_pct']}% | "
                      f"win_rate={best_row['win_rate_pct']}% | "
                      f"PF={best_row['profit_factor']} | "
                      f"maxDD={best_row['max_drawdown_pct']}% | "
                      f"trades={best_row['nb_trades']}")
            else:
                print(f"  {indicator_name:22s} -> aucune combinaison avec >= {MIN_TRADES} trades")

    full_df = pd.DataFrame(all_results)
    full_df["params"] = full_df["params"].astype(str)
    full_df.to_csv("indicator_optimization_full.csv", index=False)

    valid_df = full_df[full_df["nb_trades"] >= MIN_TRADES]
    best_df = valid_df.loc[valid_df.groupby(["asset", "indicator"])["rendement_net_pct"].idxmax()]
    best_df = best_df.sort_values(["asset", "rendement_net_pct"], ascending=[True, False])
    best_df.to_csv("indicator_optimization_best.csv", index=False)

    print(f"\nRésultats complets -> indicator_optimization_full.csv ({len(full_df)} combinaisons testées)")
    print(f"Meilleures combinaisons (>= {MIN_TRADES} trades) -> indicator_optimization_best.csv")
    print("\n=== Meilleure combinaison par actif / indicateur ===")
    print(best_df.to_string(index=False))
