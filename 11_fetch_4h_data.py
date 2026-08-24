"""
Étape 11a : Récupération des données 4h (via 1h, Coinbase n'a pas de
granularité 4h native) sur la période hors-échantillon utilisée depuis
l'étape 6, avec une marge en amont pour l'échauffement des indicateurs.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

def fetch_ohlcv_range(exchange, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    since = exchange.parse8601(start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    end_ms = exchange.parse8601(end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    step_ms = 10 * 24 * 60 * 60 * 1000  # avance de 10 jours si page vide

    all_candles = []
    while since < end_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not candles:
            since += step_ms
            continue
        candles = [c for c in candles if c[0] <= end_ms]
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    return df_1h.resample("4h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()


# (oos_start - marge, oos_end) par actif, cohérent avec les étapes 6/7/9/10
ASSETS = {
    "BTC/USD": (datetime(2023, 1, 15), datetime(2026, 7, 30), "btc_usd_4h_oos.csv"),
    "ETH/USD": (datetime(2023, 1, 15), datetime(2026, 7, 30), "eth_usd_4h_oos.csv"),
    "SOL/USD": (datetime(2023, 1, 15), datetime(2026, 7, 30), "sol_usd_4h_oos.csv"),
    "XRP/USD": (datetime(2024, 12, 15), datetime(2026, 6, 27), "xrp_usd_4h_oos.csv"),
}


if __name__ == "__main__":
    exchange = ccxt.coinbase()
    for symbol, (start_dt, end_dt, out_file) in ASSETS.items():
        print(f"\n--- Récupération 1h -> 4h pour {symbol} ({start_dt.date()} -> {end_dt.date()}) ---")
        df_1h = fetch_ohlcv_range(exchange, symbol, "1h", start_dt, end_dt)
        df_4h = resample_1h_to_4h(df_1h)
        df_4h.to_csv(out_file)
        print(f"{len(df_1h)} bougies 1h -> {len(df_4h)} bougies 4h -> {out_file}")
