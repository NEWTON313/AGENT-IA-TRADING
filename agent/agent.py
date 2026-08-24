"""
Orchestrateur de l'agent : pour chaque actif suivi, récupère les données de
marché, calcule le signal TA validé, récupère et score le sentiment des
actualités, applique les garde-fous de risque, prend une décision, et
l'applique au portefeuille de paper-trading. Aucun ordre réel n'est envoyé.
"""

import logging

from agent import config
from agent.data.market_data import get_market_data
from agent.decision.decision_engine import decide
from agent.execution.paper_broker import (
    apply_decision, current_equity, load_state, portfolio_drawdown_pct, save_state, update_peak_equity,
)
from agent.sentiment.news_fetcher import fetch_recent_news, news_for_asset
from agent.sentiment.sentiment_analyzer import score_sentiment
from agent.strategy.technical_signals import latest_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(config.AGENT_LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("agent")


def run_once() -> None:
    logger.info("=== Nouveau cycle de l'agent ===")

    state = load_state()
    all_news = fetch_recent_news()
    logger.info(f"{len(all_news)} article(s) récupéré(s) depuis les flux RSS configurés.")

    last_prices: dict[str, float] = {}
    decisions = []

    for asset in config.ASSETS:
        try:
            daily_df, weekly_df = get_market_data(asset)
        except Exception as exc:
            logger.error(f"[{asset}] Échec récupération données de marché : {exc}")
            continue

        last_prices[asset] = float(daily_df["close"].iloc[-1])
        signal = latest_signal(asset, daily_df, weekly_df)

        asset_news = news_for_asset(asset, all_news)
        sentiment = score_sentiment(asset, asset_news)

        drawdown = portfolio_drawdown_pct(state, last_prices)
        decision = decide(
            signal=signal,
            sentiment=sentiment,
            daily_df=daily_df,
            capital=current_equity(state, last_prices),
            open_positions=len(state.positions),
            portfolio_drawdown_pct=drawdown,
            has_open_position=asset in state.positions,
        )
        decisions.append(decision)

        logger.info(
            f"[{asset}] TA={signal.ta_signal} sentiment={sentiment.score:+.2f} "
            f"({sentiment.n_articles} art.) -> {decision.action} | {decision.reason}"
        )

        state = apply_decision(state, decision)

    state = update_peak_equity(state, last_prices)
    save_state(state)

    equity = current_equity(state, last_prices)
    logger.info(
        f"Équity portefeuille : {equity:.2f}$ (cash={state.cash:.2f}$, "
        f"{len(state.positions)} position(s) ouverte(s), pic={state.peak_equity:.2f}$)"
    )
    logger.info("=== Fin du cycle ===")


if __name__ == "__main__":
    run_once()
