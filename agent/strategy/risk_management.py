"""
Gestion du risque au niveau portefeuille : distance de stop ATR, sizing des
positions (risque fixe en % du capital), limite de positions concurrentes,
et circuit breaker sur le drawdown global du portefeuille.
"""

from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from agent import config


def atr_stop_distance(df: pd.DataFrame, atr_period: int = config.STOP_ATR_PERIOD,
                       multiplier: float = config.STOP_ATR_MULTIPLIER) -> float:
    """Distance de stop (en prix, pas en %) = multiplier x ATR(atr_period) au dernier jour."""
    atr = vbt.ATR.run(df["high"], df["low"], df["close"], window=atr_period).atr
    return float(multiplier * atr.iloc[-1])


def position_size(capital: float, entry_price: float, stop_distance: float,
                   risk_pct: float = config.RISK_PCT_PER_TRADE) -> float:
    """
    Sizing basé sur un risque fixe en % du capital : combien d'unités acheter
    pour que, si le stop est touché, la perte représente risk_pct du capital.
    """
    if stop_distance <= 0:
        return 0.0
    risk_amount = capital * risk_pct
    units = risk_amount / stop_distance
    max_affordable_units = capital / entry_price
    return max(0.0, min(units, max_affordable_units))


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str
    size_multiplier: float = 1.0  # appliqué par le module sentiment en aval


def check_portfolio_risk(open_positions: int, portfolio_drawdown_pct: float) -> RiskCheckResult:
    """
    Vérifie les garde-fous au niveau portefeuille avant d'autoriser une
    nouvelle entrée : nombre de positions concurrentes et circuit breaker de
    drawdown global.
    """
    if portfolio_drawdown_pct >= config.PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER:
        return RiskCheckResult(
            allowed=False,
            reason=f"Circuit breaker actif : drawdown portefeuille {portfolio_drawdown_pct:.1%} "
                   f">= seuil {config.PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER:.1%}. Nouvelles entrées bloquées.",
        )

    if open_positions >= config.MAX_CONCURRENT_POSITIONS:
        return RiskCheckResult(
            allowed=False,
            reason=f"Nombre max de positions concurrentes atteint ({open_positions}/{config.MAX_CONCURRENT_POSITIONS}).",
        )

    return RiskCheckResult(allowed=True, reason="OK")
