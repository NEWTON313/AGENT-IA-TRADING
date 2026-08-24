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
    now_ms = exchange.milliseconds()
    step_ms = 300 * 24 * 60 * 60 * 1000  # avance de 300 jours quand une page est vide (gap de listing)

    all_candles = []
    while since < now_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not candles:
            since += step_ms
            continue
        all_candles += candles
        since = candles[-1][0] + 1
        if len(candles) < 300 and candles[-1][0] + 24 * 60 * 60 * 1000 >= now_ms:
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


ASSETS_TO_FETCH = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


HISTORY_DAYS = 1825  # ~5 ans — Coinbase renverra tout ce qui est disponible (listing plus récent pour SOL/XRP)


if __name__ == "__main__":
    for symbol, output_file in ASSETS_TO_FETCH.items():
        print(f"\n--- Récupération de {symbol} ---")
        df = fetch_ohlcv_coinbase(symbol=symbol, timeframe="1d", days=HISTORY_DAYS)
        df = validate_data(df)
        df.to_csv(output_file)
        print(df.tail())
        print(f"{len(df)} bougies journalières récupérées pour {symbol} ({df.index.min().date()} -> {df.index.max().date()}) -> {output_file}")
