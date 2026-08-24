"""
Étape 1 : Pipeline de récupération des données historiques
Timeframe : Daily, période : 1 an
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_ohlcv_coinbase(symbol: str = "BTC/USD", timeframe: str = "1d", days: int = 365) -> pd.DataFrame:
    """
    Récupère l'historique OHLCV depuis Coinbase via ccxt.
    """
    exchange = ccxt.coinbase()
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if len(candles) < 300:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contrôles qualité de base : valeurs manquantes, doublons, gaps.
    """
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    missing_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D").difference(df.index)
    if len(missing_days) > 0:
        print(f"⚠️  {len(missing_days)} jour(s) manquant(s) détecté(s)")

    if df.isnull().values.any():
        print("⚠️  Valeurs manquantes détectées — vérifier avant backtest")

    return df


if __name__ == "__main__":
    df = fetch_ohlcv_coinbase(symbol="BTC/USD", timeframe="1d", days=365)
    df = validate_data(df)
    df.to_csv("btc_usd_daily_1y.csv")
    print(df.tail())
    print(f"\n{len(df)} bougies journalières récupérées.")
