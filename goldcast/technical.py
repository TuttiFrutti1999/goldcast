"""Technische Analyse: Indikatoren, Chartstruktur, Multi-Timeframe."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------ Indikatoren
def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: open/high/low/close. Alle Werte nutzen nur Daten bis zum jeweiligen Tag."""
    c, h, l = df["close"], df["high"], df["low"]
    o = pd.DataFrame(index=df.index)
    o["close"] = c
    for n in (20, 50, 100, 200):
        o[f"sma{n}"] = c.rolling(n).mean()
    for n in (20, 50, 200):
        o[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    o["macd"] = e12 - e26
    o["macd_signal"] = o["macd"].ewm(span=9, adjust=False).mean()
    o["macd_hist"] = o["macd"] - o["macd_signal"]
    o["adx"], o["pdi"], o["mdi"] = adx(df)
    # Ichimoku (ohne Vorwärtsverschiebung, damit kein Look-ahead in Features)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    o["ichi_tenkan"], o["ichi_kijun"] = tenkan, kijun
    o["ichi_span_a"] = ((tenkan + kijun) / 2).shift(26)
    o["ichi_span_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    o["rsi14"] = rsi(c)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    o["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    o["stoch_d"] = o["stoch_k"].rolling(3).mean()
    o["williams_r"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    tp = (h + l + c) / 3
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    o["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * md)
    o["roc20"] = c.pct_change(20) * 100
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    o["bb_mid"], o["bb_up"], o["bb_low"] = mid, mid + 2 * sd, mid - 2 * sd
    o["bb_pctb"] = (c - o["bb_low"]) / (o["bb_up"] - o["bb_low"])
    o["bb_width"] = (o["bb_up"] - o["bb_low"]) / mid
    o["atr14"] = atr(df)
    ke = c.ewm(span=20, adjust=False).mean()
    o["kc_up"], o["kc_low"] = ke + 2 * o["atr14"], ke - 2 * o["atr14"]
    lr = np.log(c).diff()
    o["hv20"] = lr.rolling(20).std() * np.sqrt(252) * 100
    o["hv60"] = lr.rolling(60).std() * np.sqrt(252) * 100
    return o


# ------------------------------------------------------------ Chartstruktur
def swing_points(s: pd.Series, k: int = 5):
    """Lokale Hochs/Tiefs (bestätigt nach k Bars)."""
    hi = s[(s == s.rolling(2 * k + 1, center=True).max())]
    lo = s[(s == s.rolling(2 * k + 1, center=True).min())]
    return hi, lo


def support_resistance(df: pd.DataFrame, lookback: int = 500, k: int = 5, tol: float = 0.012, top: int = 4):
    d = df.iloc[-lookback:]
    hi, lo = swing_points(d["high"], k)
    lo2 = swing_points(d["low"], k)[1]
    pts = sorted(list(hi.values) + list(lo2.values))
    if not pts:
        return [], []
    clusters, cur = [], [pts[0]]
    for p in pts[1:]:
        if p / np.mean(cur) - 1 < tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    price = d["close"].iloc[-1]
    lv = [(float(np.mean(c)), len(c)) for c in clusters]
    sup = sorted([x for x in lv if x[0] < price], key=lambda x: (-x[1], price - x[0]))[:top]
    res = sorted([x for x in lv if x[0] > price], key=lambda x: (-x[1], x[0] - price))[:top]
    fmt = lambda xs: [{"preis": round(p, 1), "beruehrungen": n} for p, n in sorted(xs, key=lambda x: x[0], reverse=True)]
    return fmt(sup), fmt(res)


def fibonacci(df: pd.DataFrame, lookback: int = 252):
    d = df.iloc[-lookback:]
    hi_i, lo_i = d["high"].idxmax(), d["low"].idxmin()
    hi, lo = d["high"].max(), d["low"].min()
    up = lo_i < hi_i  # Bewegung von Tief nach Hoch
    rng = hi - lo
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    retr = {f"{r:.1%}": round(hi - r * rng if up else lo + r * rng, 1) for r in ratios}
    ext = {f"{r:.1%}": round(hi + (r - 1) * rng if up else lo - (r - 1) * rng, 1) for r in (1.272, 1.618)}
    return {"richtung": "aufwärts" if up else "abwärts", "hoch": round(hi, 1), "tief": round(lo, 1),
            "retracements": retr, "extensions": ext}


def trend_channel(c: pd.Series, n: int = 100):
    y = np.log(c.iloc[-n:].values)
    x = np.arange(n)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sd = resid.std()
    mid = a + b * (n - 1)
    return {"steigung_pct_pro_jahr": round((np.exp(b * 252) - 1) * 100, 1),
            "mitte": round(float(np.exp(mid)), 1), "oben": round(float(np.exp(mid + 2 * sd)), 1),
            "unten": round(float(np.exp(mid - 2 * sd)), 1), "r2": round(1 - resid.var() / y.var(), 2)}


def market_structure(c: pd.Series, k: int = 5):
    hi, lo = swing_points(c, k)
    hi, lo = hi.iloc[-3:], lo.iloc[-3:]
    hh = len(hi) >= 2 and hi.iloc[-1] > hi.iloc[-2]
    hl = len(lo) >= 2 and lo.iloc[-1] > lo.iloc[-2]
    if hh and hl:
        return "Higher Highs / Higher Lows (Aufwärtsstruktur)"
    if not hh and not hl and len(hi) >= 2 and len(lo) >= 2:
        return "Lower Highs / Lower Lows (Abwärtsstruktur)"
    return "gemischt / Seitwärtsstruktur"


def candlestick_patterns(df: pd.DataFrame, n: int = 5):
    out = []
    d = df.iloc[-(n + 1):]
    for i in range(1, len(d)):
        o, h, l, c = d.iloc[i][["open", "high", "low", "close"]]
        po, pc = d.iloc[i - 1][["open", "close"]]
        body, rng = abs(c - o), h - l
        if rng <= 0:
            continue
        date = d.index[i].date().isoformat()
        if body / rng < 0.1:
            out.append({"datum": date, "muster": "Doji (Unentschlossenheit)"})
        if c > o and pc < po and c >= po and o <= pc:
            out.append({"datum": date, "muster": "Bullish Engulfing"})
        if c < o and pc > po and o >= pc and c <= po:
            out.append({"datum": date, "muster": "Bearish Engulfing"})
        lower, upper = min(o, c) - l, h - max(o, c)
        if lower > 2 * body and upper < body:
            out.append({"datum": date, "muster": "Hammer / Hanging Man"})
        if upper > 2 * body and lower < body:
            out.append({"datum": date, "muster": "Shooting Star / Inverted Hammer"})
    return out


def breakouts(df: pd.DataFrame, n: int = 20):
    c = df["close"]
    prev_hi = df["high"].rolling(n).max().shift(1)
    prev_lo = df["low"].rolling(n).min().shift(1)
    res = []
    for i in range(-10, 0):
        date = c.index[i].date().isoformat()
        if c.iloc[i] > prev_hi.iloc[i]:
            after = c.iloc[i + 1:] if i < -1 else c.iloc[0:0]
            false_b = len(after) > 0 and (after < prev_hi.iloc[i]).any()
            res.append({"datum": date, "typ": f"Ausbruch über {n}-Tage-Hoch",
                        "status": "False Breakout" if false_b else "intakt"})
        if c.iloc[i] < prev_lo.iloc[i]:
            after = c.iloc[i + 1:] if i < -1 else c.iloc[0:0]
            false_b = len(after) > 0 and (after > prev_lo.iloc[i]).any()
            res.append({"datum": date, "typ": f"Ausbruch unter {n}-Tage-Tief",
                        "status": "False Breakout" if false_b else "intakt"})
    return res


# ------------------------------------------------------------ Zusammenfassung
def _ohlc_resample(df, rule):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def timeframe_signal(df: pd.DataFrame) -> dict:
    ind = indicators(df).iloc[-1]
    c = ind["close"]
    trend = 0
    for k in ("sma20", "sma50", "sma200"):
        if pd.notna(ind.get(k)):
            trend += 1 if c > ind[k] else -1
    if pd.notna(ind["macd_hist"]):
        trend += 1 if ind["macd_hist"] > 0 else -1
    r = ind["rsi14"]
    mom = "überkauft" if r > 70 else "überverkauft" if r < 30 else "neutral"
    label = "aufwärts" if trend >= 2 else "abwärts" if trend <= -2 else "seitwärts"
    return {"trend": label, "trend_score": int(trend), "rsi": round(float(r), 1), "momentum": mom,
            "adx": round(float(ind["adx"]), 1) if pd.notna(ind["adx"]) else None}


def technical_summary(df: pd.DataFrame) -> dict:
    ind = indicators(df)
    last = ind.iloc[-1]
    sup, res = support_resistance(df)
    tfs = {"Tageschart": timeframe_signal(df)}
    for name, rule in (("Wochenchart", "W-FRI"), ("Monatschart", "ME")):
        try:
            tfs[name] = timeframe_signal(_ohlc_resample(df, rule))
        except ValueError:  # ältere pandas-Version
            tfs[name] = timeframe_signal(_ohlc_resample(df, "M"))
    ichi = "über der Wolke" if last["close"] > max(last["ichi_span_a"], last["ichi_span_b"]) else \
        "unter der Wolke" if last["close"] < min(last["ichi_span_a"], last["ichi_span_b"]) else "in der Wolke"
    r = lambda v, d=2: None if pd.isna(v) else round(float(v), d)
    return {
        "zeitebenen": tfs,
        "hinweis_intraday": "15m/1h/4h benötigen Intraday-Daten (kostenpflichtig) – im MVP nicht enthalten.",
        "indikatoren": {k: r(last[k]) for k in ["sma20", "sma50", "sma100", "sma200", "ema20", "ema50", "ema200",
                                                 "macd", "macd_signal", "macd_hist", "adx", "rsi14", "stoch_k",
                                                 "stoch_d", "cci", "williams_r", "roc20", "bb_up", "bb_mid",
                                                 "bb_low", "bb_pctb", "atr14", "kc_up", "kc_low", "hv20", "hv60"]},
        "ichimoku": ichi,
        "unterstuetzungen": sup,
        "widerstaende": res,
        "fibonacci": fibonacci(df),
        "trendkanal": trend_channel(df["close"]),
        "marktstruktur": market_structure(df["close"]),
        "candlesticks": candlestick_patterns(df),
        "breakouts": breakouts(df),
    }


def chart_payload(df: pd.DataFrame, days: int = 750) -> dict:
    ind = indicators(df).iloc[-days:]
    cols = ["close", "sma20", "sma50", "sma200", "bb_up", "bb_low", "rsi14", "macd", "macd_signal", "macd_hist"]
    out = {"datum": [d.date().isoformat() for d in ind.index]}
    for c in cols:
        out[c] = [None if pd.isna(v) else round(float(v), 2) for v in ind[c]]
    return out
