"""
Récupération des données de marché (OHLCV daily + weekly) depuis Coinbase.
Reprend la logique validée de 01_data_pipeline.py (pagination tolérante aux
trous de listing) et 08_fetch_weekly_data.py (weekly par ré-échantillonnage,
Coinbase n'a pas de granularité weekly native).
"""

from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from agent import config


def fetch_daily(symbol: str, days: int = 1825) -> pd.DataFrame:
    """
    Récupère l'historique OHLCV daily depuis Coinbase via ccxt.
    Tolère les trous de données (ex: actif temporairement délisté).
    """
    exchange = ccxt.coinbase()
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_ms = exchange.milliseconds()
    step_ms = 300 * 24 * 60 * 60 * 1000

    all_candles = []
    while since < now_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=300)
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


def to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Ré-échantillonne un DataFrame daily en bougies weekly (Coinbase n'a pas de granularité weekly native)."""
    weekly = daily_df.resample("W-MON", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return weekly


def load_or_fetch_daily(asset: str, refresh: bool = True) -> pd.DataFrame:
    """
    Charge l'historique daily depuis le CSV de la phase de recherche s'il existe,
    et le complète avec les données les plus récentes via l'API si refresh=True.
    """
    stub = config.symbol_to_filename_stub(asset)
    csv_path = config.DATA_DIR / config.DAILY_CSV_TEMPLATE.format(symbol=stub)

    if csv_path.exists():
        cached = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
    else:
        cached = pd.DataFrame()

    if not refresh:
        return cached

    fresh = fetch_daily(asset, days=30 if not cached.empty else 1825)

    if cached.empty:
        combined = fresh
    else:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    return combined


def get_market_data(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (daily_df, weekly_df) pour un actif, prêts pour le moteur de signaux."""
    daily_df = load_or_fetch_daily(asset)
    weekly_df = to_weekly(daily_df)
    return daily_df, weekly_df
