"""Feature-Engineering ohne Look-ahead-Bias.

Regeln:
* Makrodaten werden erst ab ihrem (geschätzten) Veröffentlichungsdatum verwendet
  (Referenzdatum + Verzug laut config), danach forward-fill auf Handelstage.
* COT-Daten (Stand Dienstag) erst ab Freitag (+3 Tage).
* Alle technischen Features sind kausal (rolling/ewm, nur Vergangenheit).
* Ziele (Forward-Returns) werden getrennt gehalten und nie als Feature genutzt.

Restrisiko: FRED liefert revidierte Werte (keine Echtzeit-Vintages). Revisionen
bei CPI/NFP sind klein, bei BIP größer – BIP wird daher nur als Regime-Info genutzt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import indicators


def align_published(s: pd.Series, lag_days: int, index: pd.DatetimeIndex) -> pd.Series:
    s = s.dropna().copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def build_features(px: dict, macro: pd.DataFrame, lags: dict, cot: pd.DataFrame | None,
                   etf: pd.DataFrame | None) -> pd.DataFrame:
    gold = px["gold"]
    idx = gold.index
    c = gold["close"]
    lc = np.log(c)
    ind = indicators(gold)
    f = pd.DataFrame(index=idx)

    # --- Technik / Preis
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = lc.diff(n)
    f["dist_sma50"] = c / ind["sma50"] - 1
    f["dist_sma200"] = c / ind["sma200"] - 1
    f["rsi14"] = ind["rsi14"]
    f["macd_hist_n"] = ind["macd_hist"] / c
    f["bb_pctb"] = ind["bb_pctb"]
    f["hv20"] = ind["hv20"]
    f["hv60"] = ind["hv60"]
    f["adx"] = ind["adx"]
    f["atr_n"] = ind["atr14"] / c
    f["dd_252"] = c / c.rolling(252).max() - 1
    m = idx.month
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)

    # --- Makro (veröffentlichungsbereinigt)
    M = {}
    for sid in macro.columns:
        _, lag = lags[sid]
        M[sid] = align_published(macro[sid], lag, idx)
    M = pd.DataFrame(M, index=idx)

    def has(*k):
        return all(x in M for x in k)

    if has("DFII10"):
        f["real10"] = M["DFII10"]
        f["real10_d21"] = M["DFII10"].diff(21)
        f["real10_d63"] = M["DFII10"].diff(63)
    if has("DGS10"):
        f["y10_d21"] = M["DGS10"].diff(21)
    if has("DGS2"):
        f["y2_d21"] = M["DGS2"].diff(21)
        f["y2_d63"] = M["DGS2"].diff(63)
    if has("T10Y2Y"):
        f["curve"] = M["T10Y2Y"]
    if has("T10YIE"):
        f["breakeven"] = M["T10YIE"]
        f["breakeven_d63"] = M["T10YIE"].diff(63)
    if has("DTWEXBGS"):
        f["usd_ret21"] = np.log(M["DTWEXBGS"]).diff(21)
        f["usd_ret63"] = np.log(M["DTWEXBGS"]).diff(63)
    if has("VIXCLS"):
        f["vix"] = M["VIXCLS"]
    if has("BAMLH0A0HYM2"):
        f["hy"] = M["BAMLH0A0HYM2"]
        f["hy_d63"] = M["BAMLH0A0HYM2"].diff(63)
    if has("DFF"):
        f["ff"] = M["DFF"]
        f["ff_d126"] = M["DFF"].diff(126)
    if has("ECBDFR"):
        f["ecb_d126"] = M["ECBDFR"].diff(126)
    if has("WALCL"):
        f["fedbs_yoy"] = np.log(M["WALCL"]).diff(252)
    # Monatsdaten: YoY über 12 Monate auf Originalfrequenz, dann ausgerichtet
    for sid, name in (("CPIAUCSL", "cpi_yoy"), ("CPILFESL", "core_yoy"), ("PCEPI", "pce_yoy"),
                      ("PPIACO", "ppi_yoy"), ("PAYEMS", "nfp_yoy")):
        if sid in macro:
            yoy = macro[sid].dropna().pct_change(12) * 100
            f[name] = align_published(yoy, lags[sid][1], idx)
    if "cpi_yoy" in f:
        f["cpi_yoy_d3m"] = f["cpi_yoy"] - f["cpi_yoy"].shift(63)
    if has("UNRATE"):
        f["unrate"] = M["UNRATE"]
        f["unrate_d6m"] = M["UNRATE"].diff(126)
    if has("SAHMREALTIME"):
        f["sahm"] = M["SAHMREALTIME"]

    # --- Andere Märkte (gleicher Tag = Schlusskurs, bekannt zum Feature-Zeitpunkt)
    def mkt(name):
        if name in px:
            return px[name]["close"].reindex(idx).ffill()
        return None

    s = mkt("silver")
    if s is not None:
        gsr = c / s
        f["gsr"] = gsr
        f["gsr_z"] = (gsr - gsr.rolling(252).mean()) / gsr.rolling(252).std()
    spx = mkt("spx")
    if spx is not None:
        f["spx_ret21"] = np.log(spx).diff(21)
        f["spx_ret63"] = np.log(spx).diff(63)
        f["corr_spx_63"] = lc.diff().rolling(63).corr(np.log(spx).diff())
    eur = mkt("eurusd")
    if eur is not None:
        f["eurusd_ret21"] = np.log(eur).diff(21)

    # --- Positionierung
    if cot is not None and "spec_net_oi" in cot:
        sn = align_published(cot["spec_net_oi"], 3, idx)
        f["cot_net_oi"] = sn
        f["cot_z"] = (sn - sn.rolling(156 * 5 // 3).mean()) / sn.rolling(156 * 5 // 3).std()
        f["cot_d20"] = sn.diff(20)
    if etf is not None and "tonnes" in etf:
        t = align_published(etf["tonnes"], 1, idx)
        f["etf_flow21"] = np.log(t).diff(21)
        f["etf_flow63"] = np.log(t).diff(63)

    return f.replace([np.inf, -np.inf], np.nan)


def targets(close: pd.Series, horizons: dict) -> pd.DataFrame:
    lc = np.log(close)
    return pd.DataFrame({h: lc.shift(-n) - lc for h, n in horizons.items()}, index=close.index)


FEATURE_GROUPS = {
    "Zinsen & Geldpolitik": ["real10", "real10_d21", "real10_d63", "y10_d21", "y2_d21", "y2_d63", "curve",
                             "ff", "ff_d126", "ecb_d126", "fedbs_yoy"],
    "Inflation": ["breakeven", "breakeven_d63", "cpi_yoy", "core_yoy", "pce_yoy", "ppi_yoy", "cpi_yoy_d3m"],
    "US-Dollar & Währungen": ["usd_ret21", "usd_ret63", "eurusd_ret21"],
    "Risiko & Finanzmärkte": ["vix", "hy", "hy_d63", "spx_ret21", "spx_ret63", "corr_spx_63"],
    "Konjunktur": ["unrate", "unrate_d6m", "sahm", "nfp_yoy"],
    "Positionierung & ETF": ["cot_net_oi", "cot_z", "cot_d20", "etf_flow21", "etf_flow63", "gsr", "gsr_z"],
    "Trend & Momentum": ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "dist_sma50", "dist_sma200",
                         "rsi14", "macd_hist_n", "bb_pctb", "adx", "dd_252"],
    "Volatilität": ["hv20", "hv60", "atr_n"],
    "Saisonalität": ["month_sin", "month_cos"],
}


def group_of(feature: str) -> str:
    for g, fs in FEATURE_GROUPS.items():
        if feature in fs:
            return g
    return "Sonstige"


# ------------------------------------------------------------ Regime
def regimes(f: pd.DataFrame) -> pd.DataFrame:
    """Regelbasierte, interpretierbare Marktregime (je Tag mehrere möglich)."""
    r = pd.DataFrame(index=f.index)
    g = lambda k: f[k] if k in f else pd.Series(np.nan, index=f.index)
    r["Inflationäre Phase"] = g("cpi_yoy") > 3.5
    r["Rezessionsrisiko"] = g("sahm") >= 0.5
    r["Zinssenkungszyklus"] = g("ff_d126") < -0.25
    r["Zinserhöhungszyklus"] = g("ff_d126") > 0.25
    r["Starker US-Dollar"] = g("usd_ret63") > 0.03
    r["Schwacher US-Dollar"] = g("usd_ret63") < -0.03
    r["Risk-off"] = (g("vix") > 25) | (g("hy") > 6)
    r["Risk-on"] = (g("vix") < 15) & (g("spx_ret63") > 0.03)
    r["Krisen-/Stressphase"] = (g("vix") > 35) | (g("hy_d63") > 2)
    return r.fillna(False).astype(bool)


def primary_regime(row: pd.Series) -> str:
    order = ["Krisen-/Stressphase", "Rezessionsrisiko", "Inflationäre Phase", "Zinssenkungszyklus",
             "Zinserhöhungszyklus", "Risk-off", "Starker US-Dollar", "Schwacher US-Dollar", "Risk-on"]
    for k in order:
        if row.get(k, False):
            return k
    return "Normalphase"
