"""
Étape 18 : Récupération des données pour les nouveaux actifs candidats
(ADA, DOGE, AAVE, DOT, LINK, TRUMP, FET, POL, FLOW) avant backtest de
validation. NOT/USD n'existe pas sur Coinbase — exclu.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_ohlcv_coinbase(symbol: str, timeframe: str = "1d", days: int = 1825) -> pd.DataFrame:
    exchange = ccxt.coinbase()
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_ms = exchange.milliseconds()
    step_ms = 300 * 24 * 60 * 60 * 1000

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
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    return daily_df.resample("W-MON", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()


NEW_ASSETS = {
    "ADA/USD": "ada_usd",
    "DOGE/USD": "doge_usd",
    "AAVE/USD": "aave_usd",
    "DOT/USD": "dot_usd",
    "LINK/USD": "link_usd",
    "TRUMP/USD": "trump_usd",
    "FET/USD": "fet_usd",
    "POL/USD": "pol_usd",
    "FLOW/USD": "flow_usd",
}


if __name__ == "__main__":
    for symbol, stub in NEW_ASSETS.items():
        print(f"\n--- {symbol} ---")
        daily = fetch_ohlcv_coinbase(symbol, days=1825)
        daily.to_csv(f"{stub}_daily_full.csv")
        weekly = resample_to_weekly(daily)
        weekly.to_csv(f"{stub}_weekly_full.csv")
        print(f"{len(daily)} bougies daily ({daily.index.min().date()} -> {daily.index.max().date()}), "
              f"{len(weekly)} bougies weekly")
