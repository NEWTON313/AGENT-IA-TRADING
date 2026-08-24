"""
Moteur de décision : combine le signal TA validé, le score de sentiment et
les garde-fous de risque portefeuille en une décision finale par actif.

Le signal TA reste seul décideur du BUY/SELL/HOLD. Le sentiment ne fait que
moduler la taille (réduite si modérément négatif) ou bloquer l'entrée (si très
négatif) — voir agent/sentiment/sentiment_analyzer.py pour la justification.
"""

from dataclasses import dataclass

import pandas as pd

from agent import config
from agent.strategy.risk_management import (
    RiskCheckResult, atr_stop_distance, check_portfolio_risk, position_size,
)
from agent.strategy.technical_signals import SignalState
from agent.sentiment.sentiment_analyzer import SentimentResult


@dataclass
class Decision:
    asset: str
    action: str          # "BUY", "SELL", "HOLD"
    size_units: float
    entry_price: float
    stop_price: float | None
    reason: str
    ta_signal: str
    sentiment_score: float
    sentiment_reason: str


def apply_sentiment_modulation(sentiment: SentimentResult) -> tuple[bool, float, str]:
    """
    Retourne (autorisé, multiplicateur_taille, raison) selon le score de
    sentiment, appliqué uniquement sur une entrée TA (BUY).
    """
    if sentiment.score <= config.SENTIMENT_BLOCK_THRESHOLD:
        return False, 0.0, (
            f"Entrée bloquée : sentiment très négatif ({sentiment.score:.2f}) — {sentiment.reason}"
        )
    if sentiment.score <= config.SENTIMENT_REDUCE_THRESHOLD:
        return True, config.SENTIMENT_SIZE_REDUCTION, (
            f"Taille réduite : sentiment modérément négatif ({sentiment.score:.2f}) — {sentiment.reason}"
        )
    return True, 1.0, f"Sentiment neutre/positif ({sentiment.score:.2f}) — pas de modulation."


def decide(
    signal: SignalState,
    sentiment: SentimentResult,
    daily_df: pd.DataFrame,
    capital: float,
    open_positions: int,
    portfolio_drawdown_pct: float,
    has_open_position: bool,
) -> Decision:
    if signal.ta_signal == "SELL" or (signal.ta_signal == "HOLD" and not has_open_position):
        # SELL : le signal TA a basculé baissier -> on sort si on est en position.
        # HOLD sans position ouverte : rien à faire.
        action = "SELL" if signal.ta_signal == "SELL" else "HOLD"
        return Decision(
            asset=signal.asset, action=action, size_units=0.0, entry_price=signal.close,
            stop_price=None, reason=f"Signal TA = {signal.ta_signal}",
            ta_signal=signal.ta_signal, sentiment_score=sentiment.score, sentiment_reason=sentiment.reason,
        )

    if signal.ta_signal == "HOLD" and has_open_position:
        return Decision(
            asset=signal.asset, action="HOLD", size_units=0.0, entry_price=signal.close,
            stop_price=None, reason="Position ouverte conservée (signal TA toujours haussier).",
            ta_signal=signal.ta_signal, sentiment_score=sentiment.score, sentiment_reason=sentiment.reason,
        )

    # signal.ta_signal == "BUY"
    risk_check: RiskCheckResult = check_portfolio_risk(open_positions, portfolio_drawdown_pct)
    if not risk_check.allowed:
        return Decision(
            asset=signal.asset, action="HOLD", size_units=0.0, entry_price=signal.close,
            stop_price=None, reason=risk_check.reason,
            ta_signal=signal.ta_signal, sentiment_score=sentiment.score, sentiment_reason=sentiment.reason,
        )

    sentiment_allowed, size_multiplier, sentiment_note = apply_sentiment_modulation(sentiment)
    if not sentiment_allowed:
        return Decision(
            asset=signal.asset, action="HOLD", size_units=0.0, entry_price=signal.close,
            stop_price=None, reason=sentiment_note,
            ta_signal=signal.ta_signal, sentiment_score=sentiment.score, sentiment_reason=sentiment.reason,
        )

    stop_distance = atr_stop_distance(daily_df)
    stop_price = signal.close - stop_distance
    base_size = position_size(capital, signal.close, stop_distance)
    final_size = base_size * size_multiplier

    return Decision(
        asset=signal.asset, action="BUY", size_units=final_size, entry_price=signal.close,
        stop_price=stop_price,
        reason=f"Signal TA = BUY (Supertrend haussier + tendance weekly confirmée). {sentiment_note}",
        ta_signal=signal.ta_signal, sentiment_score=sentiment.score, sentiment_reason=sentiment.reason,
    )
