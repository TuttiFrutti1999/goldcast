"""News-Analyse: Relevanzfilter, Deduplizierung, goldspezifisches Sentiment.

Regelbasiertes NLP mit domänenspezifischer Wirkungsrichtung (z. B. „Zinssenkung“ →
positiv für Gold). Bewusst transparent: jede Bewertung nennt die ausgelösten Regeln.
Optional austauschbar gegen FinBERT/LLM (siehe README).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RULES = [
    # (Regex, Score, Kategorie)
    (r"\brate cuts?\b|\bcuts? (interest )?rates\b|\bdovish\b|\beasing cycle\b|\bpivot to cuts\b", 2, "Geldpolitik"),
    (r"\brate hikes?\b|\braises? (interest )?rates\b|\bhawkish\b|\btightening\b|\bhigher for longer\b", -2, "Geldpolitik"),
    (r"\bwar\b|\binvasion\b|\bmissiles?\b|\bairstrikes?\b|\bmilitary (strike|escalation)\b|\bescalat", 2, "Geopolitik"),
    (r"\bsanctions?\b", 1, "Geopolitik"),
    (r"\bceasefire\b|\bpeace (deal|talks|agreement)\b|\btruce\b|\bde-?escalat", -1.5, "Geopolitik"),
    (r"\btariffs?\b|\btrade war\b|\bexport controls?\b", 1, "Handel"),
    (r"\btrade (deal|agreement|truce)\b", -1, "Handel"),
    (r"\brecession\b|\beconomic slowdown\b|\bcontraction\b|\bhard landing\b", 1.5, "Konjunktur"),
    (r"\bbank (failure|collapse|run)s?\b|\bbanking (crisis|turmoil)\b|\bcredit crunch\b|\bdefaults?\b|\bcontagion\b", 2, "Finanzstabilität"),
    (r"\bcentral banks? (buy|buying|bought|purchas|add|adding|added)\w*|\bgold reserves? (rise|rose|increase)", 2, "Zentralbanken"),
    (r"\bcentral banks? (sell|selling|sold)\b", -2, "Zentralbanken"),
    (r"\bsafe[- ]haven\b", 1.5, "Nachfrage"),
    (r"\b(etf|fund) inflows?\b", 1, "Nachfrage"),
    (r"\b(etf|fund) outflows?\b", -1, "Nachfrage"),
    (r"\b(physical|retail|jewel+ery) demand (rise|jump|surge|strong)", 1, "Nachfrage"),
    (r"\bweak(er)? (us )?dollar\b|\bdollar (falls?|slides?|weakens?|drops?|slumps?|retreats?)\b", 1.5, "US-Dollar"),
    (r"\bstrong(er)? (us )?dollar\b|\bdollar (rises?|rall(y|ies)|strengthens?|jumps?|gains?|firms?)\b", -1.5, "US-Dollar"),
    (r"\b(treasury )?yields? (rise|rises|jump|jumps|climb|climbs|surge|surges|spike)\b", -1.5, "Zinsen"),
    (r"\b(treasury )?yields? (fall|falls|drop|drops|slide|slides|decline|declines|ease|eases)\b", 1.5, "Zinsen"),
    (r"\b(inflation|cpi) (cools|eases|slows)\b|\bsofter[- ]than[- ]expected\b", 1, "Inflation"),
    (r"\b(inflation|cpi) (hotter|accelerat|jumps|surges)\w*|\bhotter[- ]than[- ]expected\b", -0.5, "Inflation"),
    (r"\bstrong(er)? (jobs|payrolls|labor market)\b|\bjobs? (beat|surge)\b", -1, "Arbeitsmarkt"),
    (r"\bweak(er)? (jobs|payrolls|labor market)\b|\bunemployment (rises|jumps)\b|\blayoffs\b", 1, "Arbeitsmarkt"),
]
PRICE_ACTION = [
    (r"\bgold\b.{0,40}\b(rises?|rall(y|ies)|jumps?|climbs?|gains?|surges?|soars?|record high|all-time high)\b", 1),
    (r"\bgold\b.{0,40}\b(falls?|drops?|slides?|slips?|declines?|tumbles?|plunges?|sinks?)\b", -1),
]
RELEVANT = re.compile(r"gold|bullion|xau|precious metal|fed\b|federal reserve|fomc|ecb|inflation|cpi|rate|yield|"
                      r"dollar|treasury|war|sanction|tariff|central bank|recession|bank|geopolit|payroll|jobs", re.I)
CRED = {"Federal Reserve": 1.0, "EZB": 1.0, "reuters.com": 0.95, "bloomberg.com": 0.95, "ft.com": 0.95,
        "wsj.com": 0.95, "cnbc.com": 0.9, "CNBC Economy": 0.9, "marketwatch.com": 0.9, "MarketWatch": 0.9,
        "kitco.com": 0.85, "Kitco": 0.85, "gold.org": 0.95, "investing.com": 0.8,
        "Investing.com Commodities": 0.8, "fxstreet.com": 0.75, "FXStreet Gold": 0.75}


def _norm(t: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()) - {"the", "a", "of", "to", "in", "and", "on", "for", "as"}


def dedupe(items: list[dict], thr: float = 0.6) -> list[dict]:
    items = sorted(items, key=lambda x: x["datum"], reverse=True)
    kept, sigs, seen_urls = [], [], set()
    for it in items:
        if not it.get("titel") or it.get("url") in seen_urls:
            continue
        s = _norm(it["titel"])
        if any(len(s & o) / max(1, len(s | o)) > thr for o in sigs):
            continue
        kept.append(it)
        sigs.append(s)
        seen_urls.add(it.get("url"))
    return kept


def score_item(it: dict) -> dict:
    text = f"{it.get('titel', '')}. {it.get('text', '')}"
    hits, score, cats = [], 0.0, set()
    for rx, s, cat in RULES:
        if re.search(rx, text, re.I):
            score += s
            hits.append(cat)
            cats.add(cat)
    priced = False
    for rx, s in PRICE_ACTION:
        if re.search(rx, it.get("titel", ""), re.I):
            score += s
            priced = True
            cats.add("Kursbewegung")
    score = float(np.clip(score, -5, 5))
    label = ("Stark positiv für Gold" if score >= 3 else "Leicht positiv für Gold" if score >= 1 else
             "Stark negativ für Gold" if score <= -3 else "Leicht negativ für Gold" if score <= -1 else "Neutral")
    dt = datetime.fromisoformat(it["datum"])
    age_h = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
    cred = CRED.get(it.get("quelle", ""), 0.6)
    relevance = 1.0 if re.search(r"gold|bullion|xau", text, re.I) else 0.6
    impact = abs(score) * cred * relevance
    return {**it, "score": round(score, 1), "einstufung": label, "kategorien": sorted(cats),
            "glaubwuerdigkeit": cred, "relevanz": relevance, "alter_h": round(age_h, 1),
            "bereits_eingepreist": priced or age_h > 24,
            "marktwirkung": "hoch" if impact >= 2.5 else "mittel" if impact >= 1 else "gering"}


def analyse_news(items: list[dict], max_age_h: float = 96) -> dict:
    items = [i for i in items if RELEVANT.search(f"{i.get('titel', '')} {i.get('text', '')}")]
    items = dedupe(items)
    scored = [score_item(i) for i in items]
    scored = [s for s in scored if s["alter_h"] <= max_age_h]
    if not scored:
        return {"score": None, "anzahl": 0, "nachrichten": [], "hinweis": "Keine aktuellen relevanten Nachrichten abrufbar."}
    num = den = 0.0
    for s in scored:
        w = s["glaubwuerdigkeit"] * s["relevanz"] * np.exp(-s["alter_h"] / 48) * (0.5 if s["bereits_eingepreist"] else 1)
        num += s["score"] * w
        den += w
    agg = float(np.clip(num / den / 3 * 100, -100, 100)) if den else 0.0
    geo = [s for s in scored if "Geopolitik" in s["kategorien"] and s["alter_h"] <= 72]
    cats = pd.Series([c for s in scored for c in s["kategorien"]]).value_counts().to_dict()
    pos = sorted([s for s in scored if s["score"] > 0], key=lambda s: -s["score"] * s["glaubwuerdigkeit"])[:6]
    neg = sorted([s for s in scored if s["score"] < 0], key=lambda s: s["score"] * s["glaubwuerdigkeit"])[:6]
    clean = lambda L: [{k: s[k] for k in ("titel", "url", "quelle", "datum", "score", "einstufung", "kategorien",
                                         "marktwirkung", "bereits_eingepreist", "glaubwuerdigkeit")} for s in L]
    return {"score": round(agg, 0), "anzahl": len(scored), "kategorien": cats,
            "geopolitik_meldungen_72h": len(geo),
            "geopolitik_gewicht_72h": round(sum(abs(s["score"]) * s["glaubwuerdigkeit"] for s in geo), 1),
            "top_positiv": clean(pos), "top_negativ": clean(neg),
            "nachrichten": clean(sorted(scored, key=lambda s: s["datum"], reverse=True)[:40])}


def geo_risk_index(today_weight: float, hist: pd.DataFrame | None) -> dict:
    """Perzentil des heutigen Geopolitik-Gewichts relativ zu den gespeicherten Vortagen."""
    if hist is None or len(hist) < 20:
        return {"wert": None, "hinweis": f"Index wird nach 20 Tagen Nachrichtenhistorie berechnet "
                                         f"(bisher {0 if hist is None else len(hist)})."}
    pct = float((hist["geo_gewicht"] < today_weight).mean() * 100)
    return {"wert": round(pct, 0), "hinweis": f"Perzentil gegenüber {len(hist)} gespeicherten Tagen"}


def macro_risk_index(F: pd.DataFrame) -> dict:
    """Datenbasiert: Perzentile (10 J.) von VIX, HY-Spread, Sahm, inverse Zinskurve."""
    parts = {}
    w = F.iloc[-2520:]
    for k, lbl, inv in (("vix", "VIX", False), ("hy", "High-Yield-Spread", False), ("sahm", "Sahm-Indikator", False),
                        ("curve", "Zinskurve 10Y-2Y (invertiert = Risiko)", True)):
        if k in w and w[k].notna().sum() > 200:
            s = w[k].dropna()
            p = float((s < s.iloc[-1]).mean() * 100)
            parts[lbl] = round(100 - p if inv else p, 0)
    if not parts:
        return {"wert": None, "komponenten": {}}
    return {"wert": round(float(np.mean(list(parts.values()))), 0), "komponenten": parts}
