"""
Récupération d'actualités crypto brutes via des flux RSS gratuits (aucune clé
API requise), filtrées par mot-clé pour chaque actif suivi.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import mktime

import feedparser

from agent import config


@dataclass
class NewsItem:
    title: str
    summary: str
    published: datetime
    source: str


def _parse_entry_date(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)


def fetch_recent_news(lookback_hours: int = config.NEWS_LOOKBACK_HOURS) -> list[NewsItem]:
    """Récupère les articles récents de tous les flux RSS configurés."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    items: list[NewsItem] = []

    for feed_url in config.NEWS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed_url)
        except Exception as exc:
            print(f"[news_fetcher] Erreur lecture flux {feed_url}: {exc}")
            continue

        for entry in parsed_feed.entries:
            published = _parse_entry_date(entry)
            if published is not None and published < cutoff:
                continue
            items.append(NewsItem(
                title=entry.get("title", ""),
                summary=entry.get("summary", "")[:500],
                published=published or datetime.now(timezone.utc),
                source=parsed_feed.feed.get("title", feed_url),
            ))

    return items


def news_for_asset(asset: str, all_news: list[NewsItem],
                    max_items: int = config.NEWS_MAX_ITEMS_PER_ASSET) -> list[NewsItem]:
    """Filtre les articles pertinents pour un actif donné via ses mots-clés."""
    keywords = config.ASSET_KEYWORDS.get(asset, [])
    matched = [
        item for item in all_news
        if any(kw in (item.title + " " + item.summary).lower() for kw in keywords)
    ]
    matched.sort(key=lambda item: item.published, reverse=True)
    return matched[:max_items]
