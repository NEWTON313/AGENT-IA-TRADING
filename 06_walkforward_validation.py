"""
Étape 6 : Validation Walk-Forward (out-of-sample)
Découpe l'historique en fenêtres glissantes train/test non chevauchantes.
Sur chaque fenêtre d'entraînement, on ré-optimise les paramètres RSI (comme
à l'étape 4), puis on applique ces paramètres UNIQUEMENT sur la fenêtre de
test suivante (jamais vue pendant l'optimisation). Les fenêtres de test sont
ensuite mises bout à bout pour reconstituer une performance hors-échantillon
continue, comparée à deux références sans ré-optimisation (RSI standard et
Trend+RSI) pour juger si le grid-search de l'étape 4 apportait un vrai edge
ou du surapprentissage.
"""

import itertools
import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
TRAIN_LEN = 540   # ~18 mois d'entraînement
TEST_LEN = 90     # ~3 mois de test, non vus pendant l'optimisation
STEP = TEST_LEN   # fenêtres de test contiguës, sans chevauchement
MIN_TRAIN_TRADES = 5

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


def signals_rsi(close, window, lower, upper):
    rsi = vbt.RSI.run(close, window=window).rsi
    return rsi.vbt.crossed_above(lower), rsi.vbt.crossed_below(upper)


def grid_rsi():
    for window, lower, upper in itertools.product([7, 10, 14, 21], [20, 25, 30, 35], [65, 70, 75, 80]):
        yield {"window": window, "lower": lower, "upper": upper}


def strategy_trend_filtered_rsi(close, sma_trend_w=150, rsi_w=14, lower=30, upper=70):
    sma_trend = vbt.MA.run(close, window=sma_trend_w).ma
    rsi = vbt.RSI.run(close, window=rsi_w).rsi
    uptrend = close > sma_trend
    entries = uptrend & rsi.vbt.crossed_above(lower)
    exits = rsi.vbt.crossed_below(upper) | close.vbt.crossed_below(sma_trend)
    return entries, exits


def make_folds(n, train_len=TRAIN_LEN, test_len=TEST_LEN, step=STEP):
    folds = []
    idx = 0
    while idx + train_len + test_len <= n:
        folds.append((idx, idx + train_len, idx + train_len + test_len))
        idx += step
    return folds


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
    summary_rows = []

    for asset, csv_file in ASSETS.items():
        df = load_ohlcv(csv_file)
        close = df["close"]
        n = len(close)
        folds = make_folds(n)

        if not folds:
            print(f"\n--- {asset} : historique insuffisant pour le walk-forward (n={n}) ---")
            continue

        print(f"\n--- {asset} : {len(folds)} fenêtres walk-forward (train={TRAIN_LEN}j / test={TEST_LEN}j) ---")

        wf_entries_opt = pd.Series(False, index=close.index)
        wf_exits_opt = pd.Series(False, index=close.index)
        wf_entries_vanilla = pd.Series(False, index=close.index)
        wf_exits_vanilla = pd.Series(False, index=close.index)
        wf_entries_trend = pd.Series(False, index=close.index)
        wf_exits_trend = pd.Series(False, index=close.index)

        chosen_params_log = []

        for train_start, train_end, test_end in folds:
            train_close = close.iloc[train_start:train_end]
            combined_close = close.iloc[train_start:test_end]
            test_slice_idx = combined_close.index[TRAIN_LEN:]

            # Ré-optimisation RSI sur la fenêtre d'entraînement uniquement
            best_params, best_return = None, -np.inf
            for params in grid_rsi():
                entries, exits = signals_rsi(train_close, **params)
                pf = vbt.Portfolio.from_signals(train_close, entries, exits, init_cash=INIT_CASH, fees=FEES, freq="1D")
                if pf.trades.count() >= MIN_TRAIN_TRADES and pf.total_return() > best_return:
                    best_return, best_params = pf.total_return(), params
            if best_params is None:
                best_params = {"window": 14, "lower": 30, "upper": 70}
            chosen_params_log.append(best_params)

            # Application des paramètres choisis sur la fenêtre de test (jamais vue à l'entraînement)
            entries_opt, exits_opt = signals_rsi(combined_close, **best_params)
            wf_entries_opt.loc[test_slice_idx] = entries_opt.loc[test_slice_idx]
            wf_exits_opt.loc[test_slice_idx] = exits_opt.loc[test_slice_idx]

            # Référence 1 : RSI standard fixe (14, 30, 70), jamais ré-optimisé
            entries_v, exits_v = signals_rsi(combined_close, window=14, lower=30, upper=70)
            wf_entries_vanilla.loc[test_slice_idx] = entries_v.loc[test_slice_idx]
            wf_exits_vanilla.loc[test_slice_idx] = exits_v.loc[test_slice_idx]

            # Référence 2 : stratégie combinée Trend+RSI fixe, jamais ré-optimisée
            entries_t, exits_t = strategy_trend_filtered_rsi(combined_close)
            wf_entries_trend.loc[test_slice_idx] = entries_t.loc[test_slice_idx]
            wf_exits_trend.loc[test_slice_idx] = exits_t.loc[test_slice_idx]

        oos_start = close.index[folds[0][1]]
        oos_end = close.index[folds[-1][2] - 1]
        oos_close = close.loc[oos_start:oos_end]

        variants = {
            "RSI_reoptimise_a_chaque_fenetre": (wf_entries_opt, wf_exits_opt),
            "RSI_standard_fixe_14_30_70": (wf_entries_vanilla, wf_exits_vanilla),
            "Trend(SMA150)+RSI_fixe": (wf_entries_trend, wf_exits_trend),
        }

        for variant_name, (entries_wf, exits_wf) in variants.items():
            pf = vbt.Portfolio.from_signals(
                oos_close,
                entries_wf.loc[oos_start:oos_end],
                exits_wf.loc[oos_start:oos_end],
                init_cash=INIT_CASH,
                fees=FEES,
                freq="1D",
            )
            row = {"asset": asset, "variant": variant_name, "periode_oos": f"{oos_start.date()} -> {oos_end.date()}"}
            row.update(score_portfolio(pf))
            summary_rows.append(row)

        print(f"  Paramètres RSI retenus par fenêtre : {chosen_params_log}")

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv("walkforward_validation_results.csv", index=False)

    print("\n=== Résultats Walk-Forward (hors-échantillon) ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne par variante (tous actifs confondus) ===")
    avg = (
        results_df.groupby("variant")[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]]
        .mean()
        .round(2)
        .sort_values("rendement_net_pct", ascending=False)
    )
    print(avg.to_string())
