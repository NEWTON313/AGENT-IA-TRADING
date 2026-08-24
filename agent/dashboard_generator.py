"""
Génère le tableau de bord statique (docs/index.html), régénéré à chaque
cycle de l'agent et servi via GitHub Pages — mise à jour réellement
automatique, sans dépendance à une session Claude ou à une machine
particulière. Même identité visuelle que la version précédente
(Claude Artifact), mais entièrement pilotée par les données réelles du
dernier cycle (agent/latest_cycle.json, agent/portfolio_state.json,
agent/paper_trades.csv).
"""

import csv
import json
from datetime import datetime, timezone
from html import escape

from agent import config

STYLE = """
<style>
  :root {
    --ink-950: #0e1a1d; --ink-900: #16262a;
    --paper-0: #f5f2ea; --paper-50: #ece7db;
    --line: #d8d2c2; --text: #1c2b2e; --text-dim: #5a6a68;
    --surface: #ffffff; --surface-raised: #fbfaf6;
    --accent: #2f8f86; --accent-ink: #ffffff;
    --bullish: #2f7a45; --bullish-bg: #e3efe4;
    --bearish: #b8392f; --bearish-bg: #f5e4e1;
    --amber: #93650f; --amber-bg: #f2e8d3;
    --shadow: 0 1px 2px rgba(14,26,29,.06), 0 8px 24px -12px rgba(14,26,29,.18);
    --radius: 14px; font-variant-numeric: tabular-nums;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --text: #eaf1ef; --text-dim: #9db0ac; --surface: #16262a; --surface-raised: #1b2f33;
      --line: #2a3d40; --bullish: #6fce8c; --bullish-bg: #1c3324; --bearish: #e8897d;
      --bearish-bg: #3a221f; --amber: #e0b45c; --amber-bg: #3a2f18; --accent: #4db3a8;
    }
  }
  :root[data-theme="dark"] {
    --text: #eaf1ef; --text-dim: #9db0ac; --surface: #16262a; --surface-raised: #1b2f33;
    --line: #2a3d40; --bullish: #6fce8c; --bullish-bg: #1c3324; --bearish: #e8897d;
    --bearish-bg: #3a221f; --amber: #e0b45c; --amber-bg: #3a2f18; --accent: #4db3a8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper-0); color: var(--text); font-family: "Manrope", system-ui, sans-serif; line-height: 1.5; -webkit-font-smoothing: antialiased; }
  :root:not([data-theme="light"]) body { background: var(--ink-950); }
  :root[data-theme="dark"] body { background: var(--ink-950); }
  .wrap { max-width: 960px; margin: 0 auto; padding: clamp(16px,4vw,40px) clamp(16px,4vw,32px) 64px; display: flex; flex-direction: column; gap: clamp(24px,4vw,36px); }
  header { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-mark { width: 40px; height: 40px; border-radius: 10px; background: var(--accent); color: var(--accent-ink); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .brand-mark svg { width: 22px; height: 22px; }
  h1 { font-family: "Sora", system-ui, sans-serif; font-size: clamp(22px,3.4vw,28px); font-weight: 700; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
  .tagline { margin: 2px 0 0; color: var(--text-dim); font-size: 13.5px; }
  .run-status { text-align: right; font-family: "IBM Plex Mono", monospace; font-size: 12.5px; color: var(--text-dim); line-height: 1.6; }
  .run-status strong { color: var(--text); font-weight: 500; }
  .pulse { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--bullish); margin-right: 6px; box-shadow: 0 0 0 3px var(--bullish-bg); }
  .section-label { font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-dim); margin: 0 0 12px; }
  .stat-strip { display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
  .stat { background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; }
  .stat-label { font-size: 12px; color: var(--text-dim); }
  .stat-value { font-family: "IBM Plex Mono", monospace; font-size: clamp(18px,2.6vw,22px); font-weight: 600; letter-spacing: -0.01em; }
  .stat-sub { font-size: 11.5px; color: var(--text-dim); }
  .asset-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap: 14px; }
  .asset-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 14px; }
  .asset-head { display: flex; align-items: center; justify-content: space-between; }
  .asset-name { font-family: "Sora", sans-serif; font-weight: 600; font-size: 16px; }
  .asset-price { font-family: "IBM Plex Mono", monospace; font-size: 13.5px; color: var(--text-dim); }
  .chip { display: inline-flex; align-items: center; gap: 5px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; font-weight: 500; letter-spacing: .03em; padding: 4px 9px; border-radius: 999px; text-transform: uppercase; width: fit-content; }
  .chip.sell { background: var(--bearish-bg); color: var(--bearish); }
  .chip.buy { background: var(--bullish-bg); color: var(--bullish); }
  .chip.hold { background: var(--amber-bg); color: var(--amber); }
  .chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .sentiment-row { display: flex; flex-direction: column; gap: 6px; }
  .sentiment-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-dim); }
  .sentiment-label .val { font-family: "IBM Plex Mono", monospace; color: var(--text); font-weight: 500; }
  .gauge { position: relative; height: 6px; border-radius: 999px; background: var(--paper-50); overflow: hidden; }
  :root:not([data-theme="light"]) .gauge, :root[data-theme="dark"] .gauge { background: var(--ink-900); }
  .gauge-fill { position: absolute; top: 0; bottom: 0; border-radius: 999px; }
  .gauge-mid { position: absolute; top: -3px; bottom: -3px; left: 50%; width: 1px; background: var(--line); }
  .asset-note { font-size: 12.5px; color: var(--text-dim); border-top: 1px solid var(--line); padding-top: 12px; }
  .asset-note .n { color: var(--text); }
  .log-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
  .empty-log { padding: 32px 20px; text-align: center; color: var(--text-dim); font-size: 13.5px; }
  .empty-log strong { display: block; color: var(--text); font-family: "Sora", sans-serif; font-weight: 600; font-size: 14.5px; margin-bottom: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 10px 16px; border-bottom: 1px solid var(--line); }
  th { font-family: "IBM Plex Mono", monospace; font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-dim); font-weight: 500; }
  td.num { font-family: "IBM Plex Mono", monospace; }
  .table-scroll { overflow-x: auto; }
  .strategy-panel { background: var(--surface-raised); border: 1px solid var(--line); border-radius: var(--radius); padding: 22px 24px; display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; }
  @media (max-width: 640px) { .strategy-panel { grid-template-columns: 1fr; } }
  .strategy-panel h2 { font-family: "Sora", sans-serif; font-size: 16px; margin: 0 0 8px; font-weight: 600; }
  .strategy-panel p { margin: 0; font-size: 13.5px; color: var(--text-dim); }
  .rule-list { list-style: none; margin: 12px 0 0; padding: 0; display: flex; flex-direction: column; gap: 8px; font-size: 13px; }
  .rule-list li { display: flex; gap: 8px; }
  .rule-list .k { color: var(--text-dim); min-width: 108px; flex-shrink: 0; }
  .rule-list .v { font-family: "IBM Plex Mono", monospace; }
  .metric-row { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 14px; }
  .metric { display: flex; flex-direction: column; }
  .metric .v { font-family: "IBM Plex Mono", monospace; font-size: 17px; font-weight: 600; }
  .metric .k { font-size: 11px; color: var(--text-dim); }
  footer { text-align: center; font-size: 11.5px; color: var(--text-dim); font-family: "IBM Plex Mono", monospace; }
</style>
"""

CHIP_LABELS = {"BUY": ("buy", "Buy"), "SELL": ("sell", "Sell"), "HOLD": ("hold", "Hold")}


def _fmt_usd(value: float) -> str:
    return f"{value:,.2f} $".replace(",", " ").replace(".", ",")


def _fmt_price(value: float) -> str:
    decimals = 4 if value < 10 else 2
    return f"{value:,.{decimals}f} $".replace(",", " ").replace(".", ",")


def _gauge_style(score: float) -> tuple[str, str]:
    score = max(-1.0, min(1.0, score))
    color = "var(--bullish)" if score > 0.05 else ("var(--bearish)" if score < -0.05 else "var(--text-dim)")
    if score >= 0:
        left, width = 50, score * 50
    else:
        left, width = 50 + score * 50, -score * 50
    return f"left:{left:.1f}%; width:{width:.1f}%; background:{color};", color


def _asset_card(asset: str, data: dict) -> str:
    action = data.get("action", "HOLD")
    chip_class, chip_label = CHIP_LABELS.get(action, ("hold", "Hold"))
    score = data.get("sentiment_score", 0.0)
    gauge_css, _ = _gauge_style(score)
    n_articles = data.get("n_articles", 0)
    article_word = "article" if n_articles <= 1 else "articles"

    if action == "BUY":
        note = f'<span class="n">Position ouverte</span> — {escape(data.get("reason", ""))}'
    else:
        note = f'Aucune position — <span class="n">{escape(data.get("reason", "en attente"))}</span>'

    return f"""
      <article class="asset-card">
        <div class="asset-head">
          <span class="asset-name">{escape(asset)}</span>
          <span class="asset-price">{_fmt_price(data.get("price", 0.0))}</span>
        </div>
        <span class="chip {chip_class}">Signal · {chip_label}</span>
        <div class="sentiment-row">
          <div class="sentiment-label"><span>Sentiment news ({n_articles} {article_word})</span><span class="val">{score:+.2f}</span></div>
          <div class="gauge"><div class="gauge-mid"></div><div class="gauge-fill" style="{gauge_css}"></div></div>
        </div>
        <p class="asset-note">{note}</p>
      </article>"""


def _trade_log_html(trades_csv_path) -> str:
    if not trades_csv_path.exists():
        return """
      <div class="empty-log">
        <strong>Aucun trade pour l'instant</strong>
        L'agent attend une confirmation Supertrend + tendance hebdomadaire avant d'entrer en position. C'est le comportement attendu, pas une anomalie.
      </div>"""

    with open(trades_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return """
      <div class="empty-log">
        <strong>Aucun trade pour l'instant</strong>
      </div>"""

    rows = rows[-20:][::-1]
    body_rows = "\n".join(
        f'<tr><td>{escape(r["timestamp"][:16].replace("T", " "))}</td>'
        f'<td>{escape(r["asset"])}</td>'
        f'<td>{escape(r["action"])}</td>'
        f'<td class="num">{escape(r["units"])}</td>'
        f'<td class="num">{_fmt_price(float(r["price"]))}</td>'
        f'<td class="num">{_fmt_usd(float(r["pnl"])) if r["pnl"] else "—"}</td></tr>'
        for r in rows
    )
    return f"""
      <div class="table-scroll">
        <table>
          <thead><tr><th>Date</th><th>Actif</th><th>Action</th><th>Unités</th><th>Prix</th><th>P&L</th></tr></thead>
          <tbody>{body_rows}</tbody>
        </table>
      </div>"""


def generate_dashboard() -> str:
    with open(config.PORTFOLIO_STATE_FILE, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    cycle = {"timestamp": None, "assets": {}}
    if config.LATEST_CYCLE_FILE.exists():
        with open(config.LATEST_CYCLE_FILE, "r", encoding="utf-8") as f:
            cycle = json.load(f)

    cash = portfolio.get("cash", config.INITIAL_CAPITAL)
    peak = portfolio.get("peak_equity", config.INITIAL_CAPITAL)
    positions = portfolio.get("positions", {})

    last_prices = {a: d.get("price", 0.0) for a, d in cycle.get("assets", {}).items()}
    equity = cash + sum(p["units"] * last_prices.get(a, p["entry_price"]) for a, p in positions.items())
    drawdown = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
    total_return = (equity / config.INITIAL_CAPITAL - 1) * 100

    ts_raw = cycle.get("timestamp")
    if ts_raw:
        dt = datetime.fromisoformat(ts_raw)
        last_cycle_str = dt.strftime("%d %b %Y, %H:%M UTC")
    else:
        last_cycle_str = "aucun cycle enregistré"

    asset_cards = "\n".join(_asset_card(a, cycle.get("assets", {}).get(a, {})) for a in config.ASSETS)
    trade_log_html = _trade_log_html(config.PAPER_TRADES_LOG)

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel Crypto</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22><text y=%2220%22 font-size=%2220%22>🛡️</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
{STYLE}
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand">
      <div class="brand-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 4 5v6c0 5.2 3.4 9.7 8 11 4.6-1.3 8-5.8 8-11V5l-8-3Z"/><path d="m9 12 2 2 4-4"/></svg>
      </div>
      <div>
        <h1>Sentinel</h1>
        <p class="tagline">Agent crypto — Weekly EMA10 + Daily Supertrend · paper trading uniquement</p>
      </div>
    </div>
    <div class="run-status">
      <div><span class="pulse"></span><strong>Dernier cycle</strong> — {last_cycle_str}</div>
      <div>Cycle automatique quotidien · 00:05 UTC · GitHub Actions</div>
    </div>
  </header>

  <section>
    <p class="section-label">Portefeuille (paper trading)</p>
    <div class="stat-strip">
      <div class="stat">
        <span class="stat-label">Équité totale</span>
        <span class="stat-value">{_fmt_usd(equity)}</span>
        <span class="stat-sub">{total_return:+.2f} % depuis le départ</span>
      </div>
      <div class="stat">
        <span class="stat-label">Cash disponible</span>
        <span class="stat-value">{_fmt_usd(cash)}</span>
        <span class="stat-sub">{cash / equity * 100 if equity else 0:.0f} % du portefeuille</span>
      </div>
      <div class="stat">
        <span class="stat-label">Positions ouvertes</span>
        <span class="stat-value">{len(positions)} / {config.MAX_CONCURRENT_POSITIONS}</span>
        <span class="stat-sub">max. concurrentes</span>
      </div>
      <div class="stat">
        <span class="stat-label">Pic d'équité</span>
        <span class="stat-value">{_fmt_usd(peak)}</span>
        <span class="stat-sub">drawdown {drawdown * 100:.1f} %</span>
      </div>
    </div>
  </section>

  <section>
    <p class="section-label">Actifs suivis</p>
    <div class="asset-grid">
{asset_cards}
    </div>
  </section>

  <section>
    <p class="section-label">Journal des trades simulés</p>
    <div class="log-card">{trade_log_html}
    </div>
  </section>

  <section class="strategy-panel">
    <div>
      <h2>Stratégie validée</h2>
      <p>Weekly(EMA10) + Daily Supertrend(10,3), stop suiveur ATR(14)×3, long-only. Reste en cash pendant les marchés baissiers plutôt que de shorter — validé sur 15 épisodes de baisse soutenue (voir RESEARCH_SUMMARY.md).</p>
      <ul class="rule-list">
        <li><span class="k">Entrée</span><span class="v">Supertrend haussier + close weekly &gt; EMA10</span></li>
        <li><span class="k">Sortie</span><span class="v">signal baissier OU stop ATR×3 touché</span></li>
        <li><span class="k">Sizing</span><span class="v">{config.RISK_PCT_PER_TRADE * 100:.0f} % du capital risqué / trade</span></li>
        <li><span class="k">Circuit breaker</span><span class="v">pause si drawdown ≥ {config.PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER * 100:.0f} %</span></li>
      </ul>
    </div>
    <div>
      <h2>Performance backtestée (hors-échantillon)</h2>
      <div class="metric-row">
        <div class="metric"><span class="v" style="color:var(--bullish)">+29,4 %</span><span class="k">rendement moyen</span></div>
        <div class="metric"><span class="v">56,7 %</span><span class="k">win rate</span></div>
        <div class="metric"><span class="v">2,21</span><span class="k">profit factor</span></div>
        <div class="metric"><span class="v" style="color:var(--bearish)">−27,4 %</span><span class="k">max drawdown</span></div>
      </div>
    </div>
  </section>

  <footer>
    Cycle quotidien automatique · 00:05 UTC · GitHub Actions — aucun ordre réel envoyé à Coinbase<br>
    Régénéré automatiquement à chaque cycle
  </footer>

</div>
</body>
</html>
"""


def write_dashboard() -> None:
    html = generate_dashboard()
    config.DASHBOARD_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DASHBOARD_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    write_dashboard()
    print(f"Dashboard écrit -> {config.DASHBOARD_OUTPUT_FILE}")
