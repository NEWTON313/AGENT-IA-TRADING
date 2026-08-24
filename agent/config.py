"""
Configuration centrale de l'agent : actifs suivis, paramètres de la stratégie
validée (voir RESEARCH_SUMMARY.md), paramètres de gestion du risque, chemins
de persistance et lecture des clés d'API depuis l'environnement.
"""

import os
from pathlib import Path

# --- Actifs suivis (crypto uniquement, tradables sur Coinbase) ----------------

ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]

# Mots-clés utilisés pour rattacher un article de news à un actif
ASSET_KEYWORDS = {
    "BTC/USD": ["bitcoin", "btc"],
    "ETH/USD": ["ethereum", "eth", "ether"],
    "SOL/USD": ["solana", "sol"],
    "XRP/USD": ["xrp", "ripple"],
}

# --- Paramètres de la stratégie technique validée (RESEARCH_SUMMARY.md) ------

SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
WEEKLY_TREND_EMA_WINDOW = 10
STOP_ATR_PERIOD = 14
STOP_ATR_MULTIPLIER = 3.0
FEES_PCT = 0.006  # 0.6% taker, hypothèse Coinbase Advanced standard

# --- Gestion du risque ---------------------------------------------------------

INITIAL_CAPITAL = 10_000.0
RISK_PCT_PER_TRADE = 0.02       # 2% du capital risqué par trade (sizing basé ATR)
MAX_CONCURRENT_POSITIONS = 4     # un par actif suivi
PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER = 0.25  # pause des nouvelles entrées au-delà de 25% de DD

# Sentiment : en dessous de ce score (échelle -1 à +1), une entrée TA est bloquée
SENTIMENT_BLOCK_THRESHOLD = -0.6
# Entre ce seuil et le seuil de blocage, la taille de position est réduite
SENTIMENT_REDUCE_THRESHOLD = -0.2
SENTIMENT_SIZE_REDUCTION = 0.5   # taille divisée par 2 si sentiment modérément négatif

# --- Sources de news (RSS gratuits, pas de clé requise) -----------------------

NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
NEWS_LOOKBACK_HOURS = 48
NEWS_MAX_ITEMS_PER_ASSET = 10

# --- API Claude (sentiment scoring) --------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-5"

# --- API Coinbase Advanced Trade (lecture seule pour l'instant) ---------------
# Clé CDP : jamais d'ordre réel envoyé sans confirmation explicite de l'utilisateur.

COINBASE_API_KEY = os.environ.get("COINBASE_API_KEY")
COINBASE_API_SECRET = os.environ.get("COINBASE_API_SECRET")

# --- Chemins de persistance -----------------------------------------------------

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AGENT_DIR.parent
DATA_DIR = PROJECT_DIR  # les CSV daily/weekly de la phase de recherche vivent au niveau projet
PAPER_TRADES_LOG = AGENT_DIR / "paper_trades.csv"
PORTFOLIO_STATE_FILE = AGENT_DIR / "portfolio_state.json"
AGENT_LOG_FILE = AGENT_DIR / "agent_run.log"
LATEST_CYCLE_FILE = AGENT_DIR / "latest_cycle.json"

# Dashboard statique régénéré à chaque cycle, servi via GitHub Pages (/docs sur main)
DASHBOARD_OUTPUT_FILE = PROJECT_DIR / "docs" / "index.html"

DAILY_CSV_TEMPLATE = "{symbol}_daily_full.csv"   # ex: btc_usd_daily_full.csv
WEEKLY_CSV_TEMPLATE = "{symbol}_weekly_full.csv"


def symbol_to_filename_stub(asset: str) -> str:
    """'BTC/USD' -> 'btc_usd'"""
    return asset.replace("/", "_").lower()
