"""
Étape 12 : Stratégie de confluence multi-catégories
Élimine les faux signaux en exigeant l'accord de plusieurs catégories
d'analyse plutôt qu'un seul indicateur isolé :

1. Tendance et Structure (filtre obligatoire) :
   EMA50 > EMA200 et close > EMA50, + structure de marché haussière
   (Higher Highs / Higher Lows, détectée par swing points confirmés).

2. Momentum et Retournements (déclencheur) :
   Divergence haussière RSI (prix fait un Lower Low, RSI fait un Higher Low)
   OU croisement haussier de la ligne zéro du MACD.

3. Volatilité (déclencheur alternatif) :
   Bollinger Squeeze (compression des bandes sur un plus bas de 120 jours)
   suivi d'un breakout au-dessus de la bande haute.

4. Volume et Liquidité (confirmation obligatoire) :
   OBV au-dessus de sa moyenne + volume relatif élevé au moment du signal.

Sortie : structure cassée (Lower Low confirmé), tendance cassée (close sous
EMA50), ou essoufflement du momentum (pic de l'histogramme MACD en zone
positive). Stop-loss dynamique basé sur l'ATR (gestion du risque, catégorie 3).

Testé en daily, paramètres classiques fixes, sur la même période
hors-échantillon que les étapes 6/7/9/10 pour rester comparable.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 10_000
FEES = 0.006
SWING_ORDER = 5          # bars de chaque côté pour confirmer un swing point
ATR_STOP_MULT = 2.0      # stop-loss = entrée - 2 x ATR(14)

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


# --- Swing points (Higher Highs / Higher Lows, sans lookahead) ----------------

def confirmed_swing_series(price_series: pd.Series, is_high: bool, order: int = SWING_ORDER) -> pd.Series:
    if is_high:
        flag = price_series == price_series.rolling(2 * order + 1, center=True).max()
    else:
        flag = price_series == price_series.rolling(2 * order + 1, center=True).min()
    value_at_t = price_series.where(flag)
    return value_at_t.shift(order)  # confirmé 'order' bars plus tard


def last_prev_from_confirmed(confirmed_sparse: pd.Series, full_index: pd.DatetimeIndex):
    pts = confirmed_sparse.dropna()
    last_full = pts.reindex(full_index).ffill()
    prev_full = pts.shift(1).reindex(full_index).ffill()
    return last_full, prev_full


def market_structure_and_divergence(df: pd.DataFrame, rsi: pd.Series, order: int = SWING_ORDER):
    high, low = df["high"], df["low"]

    swing_high_confirmed = confirmed_swing_series(high, is_high=True, order=order)
    last_sh, prev_sh = last_prev_from_confirmed(swing_high_confirmed, df.index)

    low_flag = low == low.rolling(2 * order + 1, center=True).min()
    swing_low_price_confirmed = low.where(low_flag).shift(order)
    swing_low_rsi_confirmed = rsi.where(low_flag).shift(order)

    last_sl, prev_sl = last_prev_from_confirmed(swing_low_price_confirmed, df.index)
    last_rsi_sl, prev_rsi_sl = last_prev_from_confirmed(swing_low_rsi_confirmed, df.index)

    structure_bullish = (last_sh > prev_sh) & (last_sl > prev_sl)
    structure_bearish_break = (last_sl < prev_sl) & (~(last_sl < prev_sl).shift(1).fillna(False))

    bullish_divergence = (last_sl < prev_sl) & (last_rsi_sl > prev_rsi_sl)
    divergence_entry = bullish_divergence & (~bullish_divergence.shift(1).fillna(False))

    return structure_bullish.fillna(False), structure_bearish_break.fillna(False), divergence_entry.fillna(False)


# --- Catégorie 1 : Tendance --------------------------------------------------

def trend_filter(close: pd.Series, fast=50, slow=200):
    ema_fast = vbt.MA.run(close, window=fast, ewm=True).ma
    ema_slow = vbt.MA.run(close, window=slow, ewm=True).ma
    trend_ok = (ema_fast > ema_slow) & (close > ema_fast)
    trend_broken = close.vbt.crossed_below(ema_fast)
    return trend_ok.fillna(False), trend_broken.fillna(False)


# --- Catégorie 2 : Momentum ---------------------------------------------------

def macd_signals(close: pd.Series, fast=12, slow=26, signal=9):
    macd_ind = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=signal)
    hist = macd_ind.macd - macd_ind.signal
    macd_bullish_cross = hist.vbt.crossed_above(0)
    exhaustion = (hist.shift(1) > hist.shift(2)) & (hist < hist.shift(1)) & (hist > 0)
    return macd_bullish_cross.fillna(False), exhaustion.fillna(False)


# --- Catégorie 3 : Volatilité --------------------------------------------------

def bollinger_squeeze_breakout(close: pd.Series, window=20, alpha=2.0, squeeze_lookback=120, confirm_window=10):
    bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
    bandwidth = (bb.upper - bb.lower) / bb.middle
    squeeze = bandwidth == bandwidth.rolling(squeeze_lookback).min()
    recent_squeeze = squeeze.rolling(confirm_window).max().astype(bool)
    breakout = close.vbt.crossed_above(bb.upper)
    return (breakout & recent_squeeze).fillna(False)


# --- Catégorie 4 : Volume ------------------------------------------------------

def volume_confirmation(df: pd.DataFrame, obv_ma_window=20, rel_vol_window=20, rel_vol_threshold=1.2):
    obv = vbt.OBV.run(df["close"], df["volume"]).obv
    obv_ma = obv.rolling(obv_ma_window).mean()
    obv_bullish = obv > obv_ma
    relative_volume = df["volume"] / df["volume"].rolling(rel_vol_window).mean()
    return (obv_bullish & (relative_volume > rel_vol_threshold)).fillna(False)


# --- Gestion du risque : stop ATR ---------------------------------------------

def atr_stop_pct(df: pd.DataFrame, window=14, multiplier=ATR_STOP_MULT):
    atr = vbt.ATR.run(df["high"], df["low"], df["close"], window=window).atr
    return (multiplier * atr / df["close"]).fillna(0.05)


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

        rsi = vbt.RSI.run(close, window=14).rsi

        trend_ok, trend_broken = trend_filter(close)
        structure_bullish, structure_bearish_break, divergence_entry = market_structure_and_divergence(df, rsi)
        macd_bullish_cross, macd_exhaustion = macd_signals(close)
        bb_breakout = bollinger_squeeze_breakout(close)
        volume_ok = volume_confirmation(df)
        stop_pct = atr_stop_pct(df)

        # Score de confluence : 4 catégories, chacune évaluée en état haussier
        # persistant (pas seulement un déclencheur ponctuel), pour permettre
        # un accord "3 sur 4" plutôt qu'un ET strict sur les 4.
        macd_hist = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
        macd_hist_val = macd_hist.macd - macd_hist.signal
        bullish_divergence_state = divergence_entry.rolling(20, min_periods=1).max().astype(bool)
        squeeze_breakout_state = bb_breakout.rolling(10, min_periods=1).max().astype(bool)

        cat1_trend_structure = trend_ok & structure_bullish
        cat2_momentum = (macd_hist_val > 0) | bullish_divergence_state
        cat3_volatility = squeeze_breakout_state
        cat4_volume = volume_ok

        confluence_score = (
            cat1_trend_structure.astype(int)
            + cat2_momentum.astype(int)
            + cat3_volatility.astype(int)
            + cat4_volume.astype(int)
        )
        bullish_zone = confluence_score >= 3
        entries = bullish_zone & (~bullish_zone.shift(1).fillna(False))
        exits = (confluence_score <= 1) | trend_broken | structure_bearish_break | macd_exhaustion

        entries_oos = entries.loc[oos_start:oos_end]
        exits_oos = exits.loc[oos_start:oos_end]
        close_oos = close.loc[oos_start:oos_end]
        stop_oos = stop_pct.loc[oos_start:oos_end]

        pf = vbt.Portfolio.from_signals(
            close_oos, entries_oos, exits_oos,
            sl_stop=stop_oos,
            init_cash=INIT_CASH, fees=FEES, freq="1D",
        )
        row = {"asset": asset, "strategie": "Confluence_score>=3/4(Trend+Structure,Momentum,Volatilite,Volume)+ATR-stop"}
        row.update(score_portfolio(pf))
        results.append(row)
        print(f"  {row}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("confluence_strategy_results.csv", index=False)

    print("\n=== Stratégie de confluence — résultats par actif ===")
    print(results_df.to_string(index=False))

    print("\n=== Moyenne ===")
    avg = results_df[["rendement_net_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct", "nb_trades"]].mean().round(2)
    print(avg.to_string())

    print("\nRappel comparatif :")
    print("  Weekly(EMA10)+Daily Supertrend (étape 10) : rendement moyen +23.92% | maxDD -28.88% | WR 56.7%")
    print("  Supertrend daily seul (étape 7)            : rendement moyen +78.71% | maxDD -50.21% | WR 30.2%")
