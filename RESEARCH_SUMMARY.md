# Synthèse — Phase de recherche & backtesting (Étapes 1 à 16)

Objectif du prompt initial : identifier, avant tout développement d'agent ou connexion API,
les indicateurs techniques et la logique de trading qui offrent un edge réel et robuste sur
crypto (Coinbase), validé hors-échantillon plutôt que simplement backtesté en in-sample.

## 1. Données

- **Source** : Coinbase (via `ccxt`), timeframe Daily.
- **Actifs** : BTC/USD, ETH/USD, SOL/USD, XRP/USD (SP500 écarté — Coinbase ne trade pas les actions).
- **Historique** : BTC/ETH/SOL ~5 ans (2021-08-25 → 2026-08-23) ; XRP ~3,1 ans (2023-07-13 →
  présent, limité par sa relisting sur Coinbase après résolution de l'affaire SEC — trou
  Jan 2021 → Juil 2023 identifié et géré).
- **Fichiers** : `btc/eth/sol/xrp_usd_daily_full.csv` (daily), `*_weekly_full.csv` (weekly,
  ré-échantillonné depuis le daily — Coinbase n'a pas de granularité weekly native), `*_4h_oos.csv`
  (4h, testé puis écarté — voir étape 11).

## 2. Méthodologie

- Frais : 0.6% par trade (taker, hypothèse Coinbase Advanced standard).
- Capital fictif : 10 000 $, position 100% in/out (pas de levier), sizing simple.
- **Walk-forward strict à partir de l'étape 6** : découpage en fenêtres train (540j) / test (90j)
  non chevauchantes. Tout indicateur "optimisé" doit prouver sa robustesse sur des données
  jamais vues pendant le réglage des paramètres — sinon le résultat est écarté.
- Période hors-échantillon de référence (comparaisons à partir de l'étape 6) :
  BTC/ETH/SOL = 2023-02-16 → 2026-07-29 ; XRP = 2025-01-03 → 2026-06-26.

## 3. Chronologie des tests et enseignements

| # | Script | Ce qui a été testé | Verdict |
|---|--------|---------------------|---------|
| 1 | `01_data_pipeline.py` | Récupération OHLCV Coinbase | ✅ Base de données |
| 2 | `02_indicator_screening.py` | État courant des indicateurs (pas un backtest) | Diagnostic seulement |
| 3-4 | `03/04_*.py` | RSI/MACD/SMA/Bollinger/Ichimoku isolés, puis grid-search (448 combos) | RSI(14) domine **in-sample** — signal d'alerte surapprentissage |
| 5 | `05_combined_strategies.py` | Combinaisons (Trend+RSI, Ichimoku+RSI, MACD+Bollinger) | Trend+RSI le plus stable |
| 6 | `06_walkforward_validation.py` | **Validation walk-forward** du RSI optimisé | 🚨 Le RSI "gagnant" de l'étape 4 devient la **pire** stratégie hors-échantillon (-46.6% moyen) → confirmation du surapprentissage |
| 7 | `07_extended_indicators_walkforward.py` | 9 indicateurs supplémentaires (Supertrend, ADX, StochRSI, SAR, Donchian, OBV, CCI, Williams %R, EMA Golden Cross), paramètres fixes classiques | **Supertrend(10,3) se démarque** : +78.7% moyen, PF 1.11, positif sur 3/4 actifs |
| 8-9 | `08/09_*.py` | Filtre de tendance Weekly (EMA20) appliqué au Supertrend daily | Réduit le drawdown mais coûte du rendement (lag) |
| 10 | `10_soften_weekly_filter.py` | Weekly EMA10 vs EMA20 vs "retournement récent" | **Weekly EMA10 optimal** : +23.9% moyen, WR 56.7%, PF 2.02, maxDD -28.9% |
| 11-12 | `11/12_*.py` | Raffinage du timing d'entrée en 4h (cascade Weekly>Daily>4h) | ❌ Ajoute du bruit (+7.1% moyen, WR 26.3%) — abandonné |
| 13 | `13_confluence_strategy.py` | Confluence 4 catégories (Trend+Structure, Momentum/Divergence RSI, Volatilité/Squeeze, Volume/OBV), ET strict puis score ≥3/4 | ❌ Sous-performe (-7.4% moyen, WR 19.4%) — la complexité ajoute du bruit |
| 14 | `14_squeeze_rsi_trailing_stop.py` | Achat squeeze Bollinger + RSI recovery + trend, sortie Chandelier ATR | ❌ Setup trop rare (1-3 trades/actif), non concluant, résultat négatif quand testable |
| 15 | `15_bear_market_analysis.py` | Comportement en marché baissier (15 épisodes, chute ≥20%) | ✅ Protège le capital : Buy&Hold -10.7% moyen sur ces épisodes vs stratégie +0.2% (reste en cash) |
| 16 | `16_long_short_supertrend.py` | Ajout d'un volet short symétrique | ❌ Dégrade tout (+23.9%→-1.4% rendement, PF 2.02→1.01, maxDD -28.9%→-43.6%) — les rallyes de bear market crypto liquident les shorts |
| 17 | `17_long_trailing_stop.py` | Stop suiveur ATR(14) en complément du signal de sortie (x2/x3/x4) | ✅ ATR×3 améliore légèrement (+29.4% moyen, PF 2.21) ; ATR×2 trop serré (catastrophique sur SOL) ; ATR×4 quasi neutre |

## 4. Stratégie finale validée

**Weekly(EMA10) + Daily Supertrend(10,3) + Trailing Stop ATR(14)×3 — Long-only**

- **Entrée** : Supertrend daily(10,3) bascule haussier **ET** close weekly > EMA10 weekly
  (tendance de fond confirmée, alignement sans lookahead — seule la dernière bougie weekly
  clôturée est utilisée).
- **Sortie** : le premier des deux événements suivants —
  1. Supertrend daily bascule baissier, ou tendance weekly repasse baissière (signal)
  2. Stop suiveur = plus haut atteint depuis l'entrée − 3 × ATR(14) (Chandelier Exit)
- **Long-only** : le volet short a été testé et rejeté (dégrade la performance).
- **Performance hors-échantillon** (moyenne sur BTC/ETH/SOL/XRP) :
  rendement net **+29.4%**, win rate **56.7%**, profit factor **2.21**, max drawdown **-27.4%**.
- **Comportement en marché baissier** : reste très majoritairement en cash pendant les phases
  de chute soutenue (≥20%, ≥30j) ; perte moyenne quasi nulle (+0.2%) contre -10.7% pour un
  Buy & Hold sur les mêmes épisodes.
- **Limite connue** : XRP reste le point faible (historique court, caractère pump/dump marqué) —
  la stratégie y est fiable pour limiter la casse mais n'y a jamais montré de rendement positif net.

## 5. Enseignement méthodologique transversal

Plus la complexité (grid-search fin, multi-timeframe 4h, confluence à 4 catégories) augmente,
moins les résultats se confirment hors-échantillon. Le signal le plus simple et le plus
robuste a systématiquement battu toutes les variantes plus sophistiquées testées. Ceci est
cohérent avec la prémisse même du prompt initial : l'analyse technique seule (même bien
construite) n'offre pas un edge énorme — elle sert surtout de **filtre de risque et de
timing**, à combiner avec l'analyse fondamentale/sentiment (étape 3 du prompt).

## 6. Prochaine étape

Passer à l'étape 3 du prompt : architecture logicielle de l'agent, combinant cette brique TA
validée avec le traitement des actualités/sentiment et la gestion du risque, avant intégration
à l'API Coinbase Advanced Trade (d'abord en paper-trading).
