"""
Client Coinbase Advanced Trade authentifié.

IMPORTANT — Garde-fou volontaire : ce module n'expose pour l'instant que des
fonctions en LECTURE SEULE (solde, ordres ouverts). Aucune fonction de
passage d'ordre réel n'est implémentée ici — ce sera ajouté séparément,
uniquement avec confirmation explicite de l'utilisateur à chaque activation,
conformément à l'étape 4 du prompt (paper-trading avant live).

Les identifiants sont lus depuis les variables d'environnement
COINBASE_API_KEY / COINBASE_API_SECRET (voir agent/config.py) — jamais en
dur dans le code, jamais affichés dans les logs.
"""

import ccxt

from agent import config


class CoinbaseCredentialsMissing(RuntimeError):
    pass


def get_authenticated_client() -> ccxt.coinbase:
    if not config.COINBASE_API_KEY or not config.COINBASE_API_SECRET:
        raise CoinbaseCredentialsMissing(
            "COINBASE_API_KEY / COINBASE_API_SECRET non configurées dans l'environnement."
        )
    return ccxt.coinbase({
        "apiKey": config.COINBASE_API_KEY,
        "secret": config.COINBASE_API_SECRET,
    })


def fetch_balance() -> dict:
    """Solde du compte Coinbase (lecture seule)."""
    client = get_authenticated_client()
    return client.fetch_balance()


def fetch_open_orders(symbol: str | None = None) -> list:
    """Ordres ouverts sur le compte (lecture seule)."""
    client = get_authenticated_client()
    return client.fetch_open_orders(symbol)


def fetch_my_trades(symbol: str, limit: int = 20) -> list:
    """Historique des trades réels passés sur le compte (lecture seule)."""
    client = get_authenticated_client()
    return client.fetch_my_trades(symbol, limit=limit)
