"""
Étape 8 : Construction des données Weekly (filtre de tendance multi-timeframe)
Coinbase ne propose pas de granularité hebdomadaire native via l'API — les
bougies weekly sont reconstruites par ré-échantillonnage des données daily
déjà récupérées (étape 1). Utilisées comme filtre de tendance de fond pour
les signaux daily (ex: Supertrend), pas comme timeframe de trading principal.
"""

import pandas as pd

ASSETS = {
    "BTC/USD": ("btc_usd_daily_full.csv", "btc_usd_weekly_full.csv"),
    "ETH/USD": ("eth_usd_daily_full.csv", "eth_usd_weekly_full.csv"),
    "SOL/USD": ("sol_usd_daily_full.csv", "sol_usd_weekly_full.csv"),
    "XRP/USD": ("xrp_usd_daily_full.csv", "xrp_usd_weekly_full.csv"),
}


def resample_to_weekly(daily_csv: str) -> pd.DataFrame:
    df = pd.read_csv(daily_csv, index_col="timestamp", parse_dates=True)
    weekly = df.resample("W-MON", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return weekly


if __name__ == "__main__":
    for symbol, (daily_csv, weekly_csv) in ASSETS.items():
        print(f"\n--- Ré-échantillonnage {symbol} (Daily -> Weekly) ---")
        weekly = resample_to_weekly(daily_csv)
        weekly.to_csv(weekly_csv)
        print(weekly.tail())
        print(f"{len(weekly)} bougies hebdomadaires ({weekly.index.min().date()} -> {weekly.index.max().date()}) -> {weekly_csv}")
