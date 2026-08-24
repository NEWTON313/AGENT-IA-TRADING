"""
Scoring du sentiment des actualités via l'API Claude. Repli neutre explicite
si ANTHROPIC_API_KEY n'est pas configurée ou si l'appel échoue, pour que le
reste de l'agent reste testable de bout en bout sans dépendance bloquante.

Rôle dans l'agent (voir plan d'architecture) : le sentiment ne génère JAMAIS
de signal d'entrée seul — il module la taille de position et peut bloquer une
entrée TA en cas d'actualité très négative. Le signal d'entrée/sortie reste
piloté à 100% par la stratégie technique validée.
"""

import json
from dataclasses import dataclass

from agent import config
from agent.sentiment.news_fetcher import NewsItem

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class SentimentResult:
    asset: str
    score: float       # -1 (très négatif) à +1 (très positif)
    confidence: float  # 0 à 1
    reason: str
    n_articles: int


def _neutral_result(asset: str, reason: str, n_articles: int = 0) -> SentimentResult:
    return SentimentResult(asset=asset, score=0.0, confidence=0.0, reason=reason, n_articles=n_articles)


def score_sentiment(asset: str, news_items: list[NewsItem]) -> SentimentResult:
    if not news_items:
        return _neutral_result(asset, "Aucune actualité récente trouvée pour cet actif.")

    if not config.ANTHROPIC_API_KEY:
        return _neutral_result(
            asset,
            "ANTHROPIC_API_KEY non configurée — sentiment neutre par défaut (repli). "
            "Définir la variable d'environnement pour activer le scoring réel.",
            n_articles=len(news_items),
        )

    if anthropic is None:
        return _neutral_result(asset, "Le package 'anthropic' n'est pas installé.", n_articles=len(news_items))

    articles_text = "\n\n".join(
        f"- [{item.source}] {item.title}\n  {item.summary}" for item in news_items
    )

    prompt = f"""Tu analyses le sentiment de marché pour {asset} à partir d'actualités récentes.

Articles :
{articles_text}

Réponds UNIQUEMENT avec un objet JSON de cette forme, sans texte autour :
{{"score": <float entre -1 et 1, -1=très négatif, 0=neutre, 1=très positif>,
  "confidence": <float entre 0 et 1>,
  "reason": "<une phrase courte expliquant le score>"}}"""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        if not text_blocks:
            raise ValueError("Aucun bloc texte dans la réponse de l'API Claude.")
        raw_text = text_blocks[0].strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()
        parsed = json.loads(raw_text)
        return SentimentResult(
            asset=asset,
            score=float(parsed["score"]),
            confidence=float(parsed["confidence"]),
            reason=str(parsed["reason"]),
            n_articles=len(news_items),
        )
    except Exception as exc:
        return _neutral_result(asset, f"Erreur API Claude ({exc}) — repli neutre.", n_articles=len(news_items))
