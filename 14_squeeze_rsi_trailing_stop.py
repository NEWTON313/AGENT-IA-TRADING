"""
Étape 13 : Achat sur compression Bollinger + RSI en survente (dans une
tendance haussière), sortie exclusivement via stop suiveur ATR (Chandelier
Exit) pour tenter de vendre au plus près du sommet.

Conditions d'achat (toutes requises) :
  - Signal haussier : close > EMA50 (tendance de fond)
  - Compression Bollinger : bandwidth dans le décile le plus bas des 120
    derniers jours (phase de faible volatilité, resserrement des bandes)
  - RSI en survente : RSI(14) croise au-dessus de 30 (sort de la survente,
    évite d'acheter alors que le prix continue de chuter)

Sortie : stop suiveur = plus haut atteint depuis l'achat − 3 × ATR(14),
ne redescend jamais (Chandelier Exit), pas d'autre condition de sortie.

Testé en daily, sur la même période hors-échantillon que les étapes
précédentes pour rester comparable.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
ATR_MULT = 3.0

OOS_BOUNDS = {
    "BTC/USD": ("2023-02-16", "2026-07-29"),
    "ETH/USD": ("2023-02-16", "2026-07-29"),
    "SOL/USD": ("2023-02-16", "2026-07-29"),
    "XRP/USD": ("2025-01-03", "2026-06-26"),
}

ASSETS = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)


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
        oos_start, oos_end = OOS_BOUNDS[asset]
        print(f"\n--- {asset} : période hors-échantillon {oos_start} -> {oos_end} ---")

        ema50 = vbt.MA.run(close, window=50, ewm=True).ma
        trend_ok = close > ema50

        bb = vbt.BBANDS.run(close, window=20, alpha=2.0)
        bandwidth = (bb.upper - bb.lower) / bb.middle
        squeeze_bar = bandwidth <= bandwidth.rolling(120).quantile(0.25)
        squeeze_recent_window = squeeze_bar.rolling(15, min_periods=1).max().astype(bool)

        rsi = vbt.RSI.run(close, window=14).rsi
        rsi_recovery = rsi.vbt.crossed_above(30)

        entries = (trend_ok & squeeze_recent_window & rsi_recovery).fillna(False)

        atr = vbt.ATR.run(df["high"], df["low"], df["close"], window=14).atr
        atr_pct = (ATR_MULT * atr / close).fillna(0.10)

        entries_oos = entries.loc[oos_start:oos_end]
        close_oos = close.loc[oos_start:oos_end]
        atr_pct_oos = atr_pct.loc[oos_start:oos_end]

        pf = vbt.Portfolio.from_signals(
            close_oos,
            entries=entries_oos,
            exits=None,
            sl_stop=atr_pct_oos,
            sl_trail=True,
            init_cash=INIT_CASH,
            fees=FEES,
            freq="1D",
        )
        row = {"asset": asset, "strategie": "BB_Squeeze+RSI_Recovery+EMA50_trend, sortie=Chandelier_ATR3"}
        row.update(score_portfolio(pf))
        results.append(row)
        print(f"  {row}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("squeeze_rsi_trailing_results.csv", index=False)

    print("\n=== Résultats par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne ===")
    avg = results_df[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]].mean().round(2)
    print(avg.to_string())

    print("\nRappel comparatif :")
    print("  Weekly(EMA10)+Daily Supertrend (étape 10) : rendement moyen +23.92% | maxDD -28.88% | WR 56.7%")
    print("  Confluence score>=3/4 (étape 12)           : rendement moyen  -7.36% | maxDD -31.21% | WR 19.4%")
