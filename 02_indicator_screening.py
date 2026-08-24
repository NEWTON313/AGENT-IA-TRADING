"""
Étape 2 : Screening d'indicateurs techniques
Calcule un jeu d'indicateurs standards (SMA, EMA, RSI, MACD, Bollinger Bands)
sur les données OHLCV récupérées à l'étape 1, et résume l'état actuel de
chaque actif (tendance, momentum, volatilité).
"""

import pandas as pd
import vectorbt as vbt


def load_ohlcv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col="timestamp", parse_dates=True)
    return df


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Calcule les indicateurs techniques usuels sur la série de clôture.
    """
    close = df["close"]

    sma_fast = vbt.MA.run(close, window=20).ma
    sma_slow = vbt.MA.run(close, window=50).ma
    ema_fast = vbt.MA.run(close, window=12, ewm=True).ma
    ema_slow = vbt.MA.run(close, window=26, ewm=True).ma

    rsi = vbt.RSI.run(close, window=14).rsi

    macd_ind = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
    macd = macd_ind.macd
    macd_signal = macd_ind.signal

    bb = vbt.BBANDS.run(close, window=20, alpha=2)

    return {
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "bb_upper": bb.upper,
        "bb_lower": bb.lower,
        "bb_middle": bb.middle,
    }


def screen_asset(symbol: str, df: pd.DataFrame, ind: dict) -> dict:
    """
    Résume l'état courant des indicateurs sous forme de signaux lisibles.
    """
    last = df.index[-1]
    close = df["close"].iloc[-1]

    trend = "haussière" if ind["sma_fast"].iloc[-1] > ind["sma_slow"].iloc[-1] else "baissière"

    rsi_val = ind["rsi"].iloc[-1]
    if rsi_val >= 70:
        rsi_zone = "surachat"
    elif rsi_val <= 30:
        rsi_zone = "survente"
    else:
        rsi_zone = "neutre"

    macd_state = "bullish" if ind["macd"].iloc[-1] > ind["macd_signal"].iloc[-1] else "bearish"

    bb_upper = ind["bb_upper"].iloc[-1]
    bb_lower = ind["bb_lower"].iloc[-1]
    bb_pct = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    return {
        "symbol": symbol,
        "date": last,
        "close": round(close, 2),
        "trend_sma20_50": trend,
        "rsi_14": round(rsi_val, 2),
        "rsi_zone": rsi_zone,
        "macd_state": macd_state,
        "bb_position_pct": round(bb_pct * 100, 1),
    }


ASSETS_TO_SCREEN = {
    "BTC/USD": "btc_usd_daily_full.csv",
    "ETH/USD": "eth_usd_daily_full.csv",
    "SOL/USD": "sol_usd_daily_full.csv",
    "XRP/USD": "xrp_usd_daily_full.csv",
}


if __name__ == "__main__":
    summaries = []
    for symbol, csv_file in ASSETS_TO_SCREEN.items():
        print(f"\n--- Screening {symbol} ---")
        df = load_ohlcv(csv_file)
        ind = compute_indicators(df)
        summary = screen_asset(symbol, df, ind)
        summaries.append(summary)
        for key, value in summary.items():
            print(f"{key}: {value}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("indicator_screening_summary.csv", index=False)
    print(f"\nRésumé sauvegardé -> indicator_screening_summary.csv")
    print(summary_df)
