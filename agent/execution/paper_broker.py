"""
Simulateur d'exécution (paper-trading local). Aucun ordre réel n'est envoyé
à Coinbase — l'intégration à l'API Advanced Trade authentifiée reste l'étape
4 du prompt, volontairement hors scope ici. Utilise les prix de marché réels
(données publiques Coinbase) pour simuler l'exécution et tient un portefeuille
virtuel persisté sur disque (CSV pour l'historique des trades, JSON pour
l'état courant).
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from agent import config
from agent.decision.decision_engine import Decision


@dataclass
class Position:
    asset: str
    units: float
    entry_price: float
    stop_price: float
    entry_date: str


@dataclass
class PortfolioState:
    cash: float = config.INITIAL_CAPITAL
    peak_equity: float = config.INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)  # asset -> Position (as dict)


def load_state() -> PortfolioState:
    if not config.PORTFOLIO_STATE_FILE.exists():
        return PortfolioState()
    with open(config.PORTFOLIO_STATE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return PortfolioState(cash=raw["cash"], peak_equity=raw["peak_equity"], positions=raw["positions"])


def save_state(state: PortfolioState) -> None:
    with open(config.PORTFOLIO_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def _log_trade(action: str, decision: Decision, units: float, price: float, pnl: float | None = None) -> None:
    is_new_file = not config.PAPER_TRADES_LOG.exists()
    with open(config.PAPER_TRADES_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "asset", "action", "units", "price", "pnl", "reason"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), decision.asset, action,
            round(units, 6), round(price, 2), round(pnl, 2) if pnl is not None else "", decision.reason,
        ])


def current_equity(state: PortfolioState, last_prices: dict[str, float]) -> float:
    equity = state.cash
    for asset, pos in state.positions.items():
        price = last_prices.get(asset, pos["entry_price"])
        equity += pos["units"] * price
    return equity


def portfolio_drawdown_pct(state: PortfolioState, last_prices: dict[str, float]) -> float:
    equity = current_equity(state, last_prices)
    peak = max(state.peak_equity, equity)
    return max(0.0, (peak - equity) / peak) if peak > 0 else 0.0


def apply_decision(state: PortfolioState, decision: Decision) -> PortfolioState:
    """Applique une décision au portefeuille virtuel (BUY/SELL/HOLD) et journalise le trade simulé."""
    if decision.action == "BUY":
        if decision.asset in state.positions:
            return state  # déjà en position, rien à faire
        cost = decision.size_units * decision.entry_price
        cost_with_fees = cost * (1 + config.FEES_PCT)
        if decision.size_units <= 0 or cost_with_fees > state.cash:
            return state
        state.cash -= cost_with_fees
        state.positions[decision.asset] = asdict(Position(
            asset=decision.asset, units=decision.size_units, entry_price=decision.entry_price,
            stop_price=decision.stop_price, entry_date=datetime.now(timezone.utc).isoformat(),
        ))
        _log_trade("BUY", decision, decision.size_units, decision.entry_price)

    elif decision.action == "SELL":
        pos = state.positions.pop(decision.asset, None)
        if pos is None:
            return state
        proceeds = pos["units"] * decision.entry_price
        proceeds_after_fees = proceeds * (1 - config.FEES_PCT)
        pnl = proceeds_after_fees - (pos["units"] * pos["entry_price"])
        state.cash += proceeds_after_fees
        _log_trade("SELL", decision, pos["units"], decision.entry_price, pnl=pnl)

    # HOLD : rien à journaliser dans le ledger de trades

    return state


def update_peak_equity(state: PortfolioState, last_prices: dict[str, float]) -> PortfolioState:
    equity = current_equity(state, last_prices)
    state.peak_equity = max(state.peak_equity, equity)
    return state
