"""KI-Interpretation: argumentiert wie ein Analyst – ausschließlich aus berechneten Daten.

Kein freier Text ohne Datenbasis: Jeder Satz wird aus konkreten Kennzahlen erzeugt.
Analyse, Prognose und Kaufumfeld-Einschätzung sind strikt getrennt.
"""
from __future__ import annotations

from datetime import date, timedelta

FMT = lambda x: f"{x:,.0f}".replace(",", ".")


def _de(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def _pct(x, d=1):
    return f"{'+' if x >= 0 else ''}{x:.{d}f} %".replace(".", ",")


def analysis_text(price: dict, tech: dict, fund: dict, fc: dict, senti: dict, regimes: list[str],
                  analog: dict) -> list[str]:
    p = []
    d, w = tech["zeitebenen"]["Tageschart"], tech["zeitebenen"]["Wochenchart"]
    trend_txt = {"aufwärts": "Aufwärtstrend", "abwärts": "Abwärtstrend", "seitwärts": "Seitwärtsphase"}
    s = (f"Gold notiert bei {FMT(price['usd'])} USD ({FMT(price['eur'])} EUR) je Feinunze, "
         f"{_pct(price['veraenderung_1m_pct'])} gegenüber dem Vormonat. ") if price.get("eur") else \
        f"Gold notiert bei {FMT(price['usd'])} USD je Feinunze, {_pct(price['veraenderung_1m_pct'])} zum Vormonat. "
    s += (f"Auf Tagesbasis befindet sich der Kurs in einem {trend_txt[d['trend']]}, im Wochenchart in einem "
          f"{trend_txt[w['trend']]}; die Marktstruktur zeigt {tech['marktstruktur']}.")
    p.append(s)

    drivers = []
    r = fund.get("realrendite_10j")
    if r and r.get("veraenderung_1m") is not None:
        ch = r["veraenderung_1m"]
        if abs(ch) >= 0.1:
            drivers.append(f"die US-Realrenditen sind im Monatsvergleich um {_de(abs(ch))} Prozentpunkte "
                           f"{'gestiegen – das belastet' if ch > 0 else 'gesunken – das stützt'} Gold")
    u = fund.get("usd_index")
    if u and u.get("veraenderung_1m_pct") is not None and abs(u["veraenderung_1m_pct"]) >= 1:
        ch = u["veraenderung_1m_pct"]
        drivers.append(f"der US-Dollar hat {_pct(ch)} {'aufgewertet (Gegenwind)' if ch > 0 else 'abgewertet (Rückenwind)'}")
    if senti.get("score") is not None and abs(senti["score"]) >= 20:
        drivers.append(f"das Nachrichten-Sentiment ist {'positiv' if senti['score'] > 0 else 'negativ'} "
                       f"({senti['score']:+.0f} von ±100)")
    if regimes:
        drivers.append("das erkannte Marktregime lautet " + ", ".join(regimes))
    if drivers:
        p.append("Wesentliche Einflüsse: " + "; ".join(drivers) + ".")

    ind = tech["indikatoren"]
    warn = []
    if ind.get("rsi14") and ind["rsi14"] > 70:
        warn.append(f"der RSI (14) signalisiert mit {ind['rsi14']:.0f} eine überkaufte Lage – das Risiko einer "
                    f"technischen Korrektur steigt")
    elif ind.get("rsi14") and ind["rsi14"] < 30:
        warn.append(f"der RSI (14) zeigt mit {ind['rsi14']:.0f} eine überverkaufte Lage – technische Gegenbewegungen "
                    f"werden wahrscheinlicher")
    if ind.get("sma200") and price["usd"] / ind["sma200"] - 1 > 0.15:
        warn.append(f"der Kurs liegt {(price['usd'] / ind['sma200'] - 1) * 100:.0f} % über dem 200-Tage-Durchschnitt "
                    f"(historisch hohe Ausdehnung)")
    if ind.get("bb_pctb") is not None and ind["bb_pctb"] > 1:
        warn.append("der Kurs notiert oberhalb des oberen Bollinger-Bands")
    if warn:
        p.append("Allerdings: " + "; ".join(warn) + ".")

    f30, f90 = fc.get("30d"), fc.get("90d")
    if f30 and f90:
        tone = lambda q: ("überwiegend positiv" if q >= 60 else "leicht positiv" if q >= 53 else
                          "überwiegend negativ" if q <= 40 else "leicht negativ" if q <= 47 else "ausgeglichen")
        p.append(f"Die Modelle bewerten die kommenden 30 Tage als {tone(f30['wahrscheinlichkeit_steigend'])} "
                 f"(P(steigend) {f30['wahrscheinlichkeit_steigend']:.0f} %, erwartete Veränderung "
                 f"{_pct(f30['veraenderung_pct'])}, 80-%-Spanne {FMT(f30['intervall_80'][0])}–{FMT(f30['intervall_80'][1])} USD). "
                 f"Für 90 Tage: {tone(f90['wahrscheinlichkeit_steigend'])} ({f90['wahrscheinlichkeit_steigend']:.0f} %). "
                 f"Prognosekonfidenz: {f30['konfidenz']}.")
        if f30["treiber"]:
            t = f30["treiber"][:3]
            p.append("Stärkste Modelltreiber (30 Tage): " + ", ".join(
                f"{x['faktor']} ({'+' if x['beitrag_pct'] > 0 else ''}{_de(x['beitrag_pct'])} Pp.)" for x in t) + ".")
    if analog.get("median_3M_pct") is not None:
        p.append(f"Historische Analogien mit ähnlichem Makro-/Marktumfeld zeigten danach im Median "
                 f"{_pct(analog['median_3M_pct'])} über 3 Monate ({analog['anteil_steigend']:.0f} % der Fälle steigend).")
    return p


def buy_environment(fc: dict, tech: dict, price: dict, cfg: dict) -> dict:
    """Einschätzung des Kaufumfelds – KEINE Anlageempfehlung."""
    h = cfg["decision"]["min_horizon"]
    f = fc.get(h)
    if not f:
        return {"einschaetzung": "keine Daten"}
    cost = cfg["decision"]["cost_etc_pct"]
    cost_phys = cfg["decision"]["spread_physical_pct"]
    s, why = 0, []
    exp = f["veraenderung_pct"]
    if exp - cost > 2:
        s += 1; why.append(f"erwartete {h}-Rendite ({_pct(exp)}) übersteigt Kosten ({_de(cost, 1)} %) deutlich")
    elif exp < cost:
        s -= 1; why.append(f"erwartete {h}-Rendite ({_pct(exp)}) deckt die Kosten ({_de(cost, 1)} %) kaum")
    pu = f["wahrscheinlichkeit_steigend"]
    if pu >= 60:
        s += 1; why.append(f"P(steigend, {h}) = {pu:.0f} %")
    elif pu <= 45:
        s -= 1; why.append(f"P(steigend, {h}) nur {pu:.0f} %")
    rsi = tech["indikatoren"].get("rsi14")
    if rsi and rsi > 70:
        s -= 1; why.append("kurzfristig überkauft (RSI > 70)")
    elif rsi and rsi < 35:
        s += 1; why.append("kurzfristig überverkauft (RSI < 35)")
    sma = tech["indikatoren"].get("sma200")
    if sma:
        ext = price["usd"] / sma - 1
        if ext > 0.15:
            s -= 1; why.append(f"{ext * 100:.0f} % über SMA 200 – erhöhtes Rückschlagrisiko")
        elif ext < -0.03:
            s += 1; why.append("Kurs unter SMA 200 – günstigeres Einstiegsniveau relativ zum Trend")
    label = "eher günstig" if s >= 2 else "eher ungünstig" if s <= -2 else "neutral"
    return {"einschaetzung": label, "score": s, "gruende": why,
            "kosten_hinweis": f"Physisches Gold: typ. {_de(cost_phys, 1)} % Spread (Break-even nötig), ETC/Xetra-Gold: ca. {_de(cost, 1)} %. "
                              f"EUR-Anleger tragen zusätzlich das EUR/USD-Wechselkursrisiko.",
            "disclaimer": "Statistische Einschätzung des Marktumfelds, keine Anlageberatung oder Kaufempfehlung."}


def daily_qa(price, tech, fc, senti, events, buy, fund, regimes) -> list[dict]:
    q = []
    q.append({"frage": "Wie entwickelt sich der Goldpreis aktuell?",
              "antwort": f"{FMT(price['usd'])} USD; 1 Tag {_pct(price['veraenderung_1t_pct'])}, 1 Woche "
                         f"{_pct(price['veraenderung_1w_pct'])}, 1 Monat {_pct(price['veraenderung_1m_pct'])}. "
                         f"Trend Tageschart: {tech['zeitebenen']['Tageschart']['trend']}, Wochenchart: "
                         f"{tech['zeitebenen']['Wochenchart']['trend']}."})
    f30 = fc.get("30d", {})
    drv = ", ".join(f"{d['faktor']} ({d['richtung']})" for d in f30.get("treiber", [])[:4])
    q.append({"frage": "Warum steigt oder fällt er?",
              "antwort": f"Größte Modellbeiträge: {drv or 'keine Daten'}. Regime: {', '.join(regimes) or 'Normalphase'}."})
    tops = (senti.get("top_positiv") or [])[:2] + (senti.get("top_negativ") or [])[:2]
    q.append({"frage": "Welche Nachrichten sind verantwortlich?",
              "antwort": "; ".join(f"{t['titel']} ({t['quelle']}, {t['datum'][:10]}, {t['einstufung']})" for t in tops)
              or "Keine relevanten Nachrichten mit klarer Wirkung gefunden."})
    q.append({"frage": "Was erwarten die Modelle?",
              "antwort": " | ".join(f"{h}: " + ", ".join(f"{m} {_pct(v['veraenderung_pct'])}" for m, v in f["modelle"].items())
                                    for h, f in fc.items() if h in ("30d", "90d"))})
    q.append({"frage": "Wie wahrscheinlich ist ein steigender Goldpreis?",
              "antwort": ", ".join(f"{h}: {f['wahrscheinlichkeit_steigend']:.0f} %" for h, f in fc.items() if not h.startswith("_"))})
    q.append({"frage": "Welche Kursziele sind plausibel (80-%-Spanne)?",
              "antwort": "; ".join(f"{h}: {FMT(f['intervall_80'][0])}–{FMT(f['intervall_80'][1])} USD (Median {FMT(f['median_usd'])})"
                                   for h, f in fc.items() if h in ("7d", "30d", "90d", "365d"))})
    soon = [e for e in events if date.fromisoformat(e["datum"]) <= date.today() + timedelta(days=21)][:6]
    q.append({"frage": "Welche Ereignisse könnten die Prognose verändern?",
              "antwort": ("; ".join(f"{e['datum']}: {e['ereignis']}" for e in soon) or "Keine geladenen Termine in 3 Wochen")
              + ". Dazu jederzeit: geopolitische Eskalationen, Überraschungen bei Inflation/Arbeitsmarkt, Zentralbankkäufe."})
    q.append({"frage": "Ist die Lage günstig, neutral oder ungünstig für einen Goldkauf?",
              "antwort": f"{buy['einschaetzung']} – " + "; ".join(buy.get("gruende", [])) + f". {buy.get('disclaimer', '')}"})
    return q


EVENT_IMPACT = {
    "Zentralbank": "Hoch: Zinssignal bewegt Realrenditen und US-Dollar; dovish = tendenziell positiv für Gold.",
    "Inflation": "Mittel–hoch: Über Erwartung → höhere Zinserwartung (kurzfristig negativ), unter Erwartung → positiv.",
    "Arbeitsmarkt": "Hoch: Schwache Daten → Zinssenkungserwartung steigt (positiv), starke Daten → negativ.",
}
