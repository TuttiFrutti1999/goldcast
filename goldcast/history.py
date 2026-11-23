"""Historische Analyse: Statistik, Saisonalität, Krisen, Zinszyklen, Korrelationen, Analogien."""
from __future__ import annotations

import numpy as np
import pandas as pd

CRISES = [  # historische Ereignisse (Startdatum)
    ("1990-08-02", "Irak-Invasion Kuwait"),
    ("1997-10-27", "Asienkrise (Mini-Crash)"),
    ("2001-09-11", "Anschläge 11. September"),
    ("2008-09-15", "Lehman-Pleite"),
    ("2010-05-06", "Euro-Schuldenkrise / Flash Crash"),
    ("2011-08-05", "US-Rating-Herabstufung"),
    ("2013-04-12", "Gold-Crash / Taper-Erwartung"),
    ("2016-06-24", "Brexit-Votum"),
    ("2020-03-09", "Covid-Crash"),
    ("2022-02-24", "Russland-Invasion Ukraine"),
    ("2023-03-10", "SVB-Bankenkrise"),
    ("2023-10-09", "Nahost-Krieg (Okt. 2023)"),
]


def _fwd(c: pd.Series, d: pd.Timestamp, n: int):
    i = c.index.searchsorted(d)
    if i + n >= len(c) or i >= len(c):
        return None
    return round((c.iloc[i + n] / c.iloc[i - 1] - 1) * 100, 1) if i > 0 else None


def long_term_stats(c: pd.Series) -> dict:
    lr = np.log(c).diff().dropna()
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    dd = c / c.cummax() - 1
    yearly = c.resample("YE").last().pct_change().dropna() * 100 if hasattr(c, "resample") else None
    w52 = c.iloc[-252:]
    return {
        "zeitraum": [c.index[0].date().isoformat(), c.index[-1].date().isoformat()],
        "allzeithoch": {"preis": round(float(c.max()), 1), "datum": c.idxmax().date().isoformat()},
        "allzeittief_zeitraum": {"preis": round(float(c.min()), 1), "datum": c.idxmin().date().isoformat()},
        "hoch_52w": round(float(w52.max()), 1), "tief_52w": round(float(w52.min()), 1),
        "rendite_pa_pct": round(((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
        "volatilitaet_pa_pct": round(float(lr.std() * np.sqrt(252) * 100), 1),
        "max_drawdown_pct": round(float(dd.min() * 100), 1),
        "max_drawdown_datum": dd.idxmin().date().isoformat(),
        "aktueller_drawdown_pct": round(float(dd.iloc[-1] * 100), 1),
        "jahresrenditen": {str(k.year): round(float(v), 1) for k, v in yearly.items()} if yearly is not None else {},
    }


def seasonality(c: pd.Series) -> list[dict]:
    m = c.resample("ME").last().pct_change().dropna() * 100
    names = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    g = m.groupby(m.index.month)
    return [{"monat": names[k - 1], "mittel_pct": round(float(v.mean()), 2), "median_pct": round(float(v.median()), 2),
             "anteil_positiv": round(float((v > 0).mean() * 100), 0), "n": int(len(v))} for k, v in g]


def crisis_reactions(c: pd.Series) -> list[dict]:
    out = []
    for d, name in CRISES:
        t = pd.Timestamp(d)
        if t < c.index[0]:
            continue
        out.append({"datum": d, "ereignis": name, "1M_pct": _fwd(c, t, 21), "3M_pct": _fwd(c, t, 63),
                    "6M_pct": _fwd(c, t, 126), "12M_pct": _fwd(c, t, 252)})
    return out


def rate_cycles(c: pd.Series, ff: pd.Series | None) -> list[dict]:
    """Erste Zinssenkung / -erhöhung nach Pause (aus Fed-Funds-Daten abgeleitet)."""
    if ff is None or ff.dropna().empty:
        return []
    m = ff.dropna().resample("ME").mean()
    ch = m.diff(3)
    events, last_seen = [], {}
    for d, v in ch.items():
        kind = "Erste Zinssenkung" if v <= -0.2 else "Erste Zinserhöhung" if v >= 0.2 else None
        if not kind:
            continue
        if kind not in last_seen or (d - last_seen[kind]).days > 365:
            events.append((d, kind))  # Beginn eines neuen Zyklus
        last_seen[kind] = d
    out = []
    for d, kind in events:
        if d < c.index[0]:
            continue
        out.append({"datum": d.date().isoformat(), "ereignis": kind, "3M_pct": _fwd(c, d, 63),
                    "6M_pct": _fwd(c, d, 126), "12M_pct": _fwd(c, d, 252)})
    return out


def recession_behaviour(c: pd.Series, sahm: pd.Series | None) -> dict | None:
    if sahm is None or sahm.dropna().empty:
        return None
    s = sahm.dropna().resample("ME").last()
    rec = s >= 0.5
    m = c.resample("ME").last().pct_change() * 100
    m = m.reindex(s.index)
    return {"mittel_monat_rezessionssignal_pct": round(float(m[rec].mean()), 2),
            "mittel_monat_sonst_pct": round(float(m[~rec].mean()), 2), "monate_signal": int(rec.sum())}


def correlations(px: dict, macro: pd.DataFrame) -> list[dict]:
    g = np.log(px["gold"]["close"]).diff()
    pairs = {}
    for k, lbl in (("silver", "Silber"), ("spx", "S&P 500"), ("eurusd", "EUR/USD")):
        if k in px:
            pairs[lbl] = np.log(px[k]["close"]).diff()
    if "DTWEXBGS" in macro:
        pairs["US-Dollar-Index"] = np.log(macro["DTWEXBGS"].dropna()).diff()
    if "DFII10" in macro:
        pairs["10Y-Realrendite (Δ)"] = macro["DFII10"].dropna().diff()
    if "DGS10" in macro:
        pairs["10Y-Rendite (Δ)"] = macro["DGS10"].dropna().diff()
    out = []
    for lbl, s in pairs.items():
        d = pd.concat([g, s], axis=1, join="inner").dropna()
        d = d.resample("W-FRI").sum()  # Wochenbasis gegen Zeitzonen-Effekte
        row = {"anlage": lbl}
        for yrs in (1, 3, 10):
            x = d.iloc[-52 * yrs:]
            row[f"{yrs}J"] = round(float(x.iloc[:, 0].corr(x.iloc[:, 1])), 2) if len(x) > 20 else None
        out.append(row)
    return out


ANALOG_FEATURES = ["real10", "real10_d63", "usd_ret63", "cpi_yoy", "ff_d126", "vix", "ret_63", "dist_sma200", "hv60"]


def analogs(F: pd.DataFrame, close: pd.Series, k: int = 8, horizon: int = 63) -> dict:
    cols = [c for c in ANALOG_FEATURES if c in F]
    X = F[cols].dropna()
    if len(X) < 500:
        return {"hinweis": "zu wenig Daten"}
    mu, sd = X.mean(), X.std()
    Z = (X - mu) / sd
    now = Z.iloc[-1]
    hist = Z.iloc[:-(horizon + 252)]  # nur abgeschlossene Episoden, Abstand zum heute
    dist = np.sqrt(((hist - now) ** 2).sum(axis=1))
    picks = []
    for d in dist.sort_values().index:
        if all(abs((d - p).days) > 120 for p in picks):
            picks.append(d)
        if len(picks) == k:
            break
    rows, fw = [], []
    for d in picks:
        r = _fwd(close, d + pd.Timedelta(days=1), horizon)
        fw.append(r)
        rows.append({"datum": d.date().isoformat(), "aehnlichkeit": round(float(1 / (1 + dist[d])), 2),
                     "goldpreis_danach_3M_pct": r})
    fw = [x for x in fw if x is not None]
    return {"merkmale": cols, "episoden": rows,
            "median_3M_pct": round(float(np.median(fw)), 1) if fw else None,
            "anteil_steigend": round(float(np.mean([x > 0 for x in fw]) * 100), 0) if fw else None}


def historical_analysis(px, macro, F) -> dict:
    c = px["gold"]["close"]
    return {"statistik": long_term_stats(c), "saisonalitaet": seasonality(c), "krisen": crisis_reactions(c),
            "zinszyklen": rate_cycles(c, macro.get("DFF")), "rezession": recession_behaviour(c, macro.get("SAHMREALTIME")),
            "korrelationen": correlations(px, macro), "analogien": analogs(F, c)}
