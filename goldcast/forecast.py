"""Aktuelle Prognose: Modelle auf allen realisierten Daten, Ensemble, Monte Carlo, Szenarien."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import usable_columns
from .features import group_of
from .models import ARIMA, FEATURE_MODELS, GARCH, GBM, GBMDirection


def model_weights(bt: pd.DataFrame | None, h: int | str, models: list[str], regime: str,
                  window: int = 750, shrink: float = 0.5) -> tuple[pd.Series, str]:
    """Inverse-MSE-Gewichte aus Out-of-Sample-Fehlern; bei genug Daten regimeabhängig."""
    eq = pd.Series(1 / len(models), index=models)
    if bt is None or bt.empty:
        return eq, "Gleichgewichtung (noch kein Backtest)"
    d = bt[(bt["horizont"] == h) & bt["modell"].isin(models)].dropna(subset=["ist"])
    if d.empty:
        return eq, "Gleichgewichtung"
    d["e2"] = (d["prognose"] - d["ist"]) ** 2
    recent = d[d["pos"] > d["pos"].max() - window]
    inv = 1 / recent.groupby("modell")["e2"].mean().reindex(models)
    w = (inv / inv.sum()).fillna(0)
    basis = "Out-of-Sample-Fehler der letzten ~3 Jahre"
    reg = d[d["regime"] == regime]
    if reg["pos"].nunique() >= 200:
        inv_r = 1 / reg.groupby("modell")["e2"].mean().reindex(models)
        w = 0.5 * w + 0.5 * (inv_r / inv_r.sum()).fillna(0)
        basis += f" + Fehler im Regime „{regime}“"
    w = shrink * w + (1 - shrink) * eq
    return w / w.sum(), basis


def _confidence(spread_models: float, sigma: float, hit: float | None, base: float | None) -> tuple[str, int]:
    """Modell-Einigkeit + Richtungsgüte ÜBER der naiven Basis („steigt immer“)."""
    agree = max(0.0, 1 - spread_models / (0.5 * sigma)) if sigma > 0 else 0
    edge = None if hit is None else hit - max(base or 50, 100 - (base or 50))
    skill = 0.25 if edge is None else min(1.0, max(0.0, edge / 10))
    score = int(round(100 * (0.5 * agree + 0.5 * skill)))
    return ("hoch" if score >= 65 else "mittel" if score >= 40 else "niedrig"), score


def current_forecast(F, Y, rets, close, eurusd, horizons, cfg, regime_now, bt=None, met=None,
                     direction=None) -> dict:
    rng = np.random.default_rng(cfg["models"]["random_seed"])
    mstart = pd.Timestamp(cfg["general"]["model_start"])
    lo = F.index.searchsorted(mstart)
    x_now = F.iloc[[-1]]
    p0 = float(close.iloc[-1])
    fx = float(eurusd) if eurusd else None
    arima = ARIMA().fit_series(rets)
    garch = GARCH().fit_series(rets)
    out = {}
    for h, n in horizons.items():
        ok = Y[h].iloc[lo:].notna()
        Xtr, ytr = F.iloc[lo:][ok], Y[h].iloc[lo:][ok]
        cols = usable_columns(Xtr)
        preds, contrib = {}, None
        for M in FEATURE_MODELS:
            m = M(seed=cfg["models"]["random_seed"])
            m.fit(Xtr[cols], ytr)
            preds[M.name] = float(m.predict(x_now[cols])[0])
            if isinstance(m, GBM):
                contrib = pd.Series(m.contributions(x_now[cols])[0][:-1], index=cols)
        preds["ARIMA"] = arima.forecast_cum(n)
        p_up_clf = float(GBMDirection(cfg["models"]["random_seed"]).fit(Xtr[cols], ytr).predict_proba(x_now[cols])[0])

        w, basis = model_weights(bt, h, list(preds), regime_now)
        mu = float(sum(preds[k] * w[k] for k in preds))
        sigma = garch.sigma_h(n)

        # Monte Carlo: Bootstrap der standardisierten OOS-Residuen (empirisch kalibriert, fette Ränder)
        z = None
        if bt is not None and not bt.empty:
            z = bt[(bt["horizont"] == h) & (bt["modell"] == "Ensemble")]["z"].dropna().values
        if z is None or len(z) < 200:
            z = rng.standard_t(5, cfg["models"]["mc_paths"]) * np.sqrt(3 / 5)
            mc_basis = "t-Verteilung (noch zu wenig Backtest-Residuen)"
        else:
            z = rng.choice(z, cfg["models"]["mc_paths"])
            mc_basis = f"Bootstrap aus {len(bt[(bt['horizont'] == h) & (bt['modell'] == 'Ensemble')]['z'].dropna())} OOS-Residuen"
        sim = mu + sigma * z
        prices = p0 * np.exp(sim)
        q = lambda a: float(np.quantile(prices, a))
        thr = 0.5 * sigma
        bull, bear = sim > thr, sim < -thr
        neu = ~bull & ~bear
        # Klassifikator nur einbeziehen, wenn er im Backtest besser kalibriert war als die Basisrate
        w_clf, clf_note = 0.0, "nicht genutzt (im Backtest nicht besser als Basisrate)"
        dq = (direction or {}).get(h)
        if dq and dq["brier"] < dq["brier_basis"]:
            w_clf, clf_note = 0.5, f"genutzt (Brier {dq['brier']} < Basis {dq['brier_basis']})"
        p_up = (1 - w_clf) * float((sim > 0).mean()) + w_clf * p_up_clf

        hit = base = None
        if met is not None and not met.empty:
            r = met[(met["horizont"] == h) & (met["modell"] == "Ensemble")]
            if len(r):
                hit, base = float(r["trefferquote_richtung"].iloc[0]), float(r["anteil_steigend"].iloc[0])
        spread = float(np.std(list(preds.values())))
        conf, conf_score = _confidence(spread, sigma, hit, base)

        drivers = []
        if contrib is not None:
            g = contrib.groupby(contrib.index.map(group_of)).sum()
            tot = g.abs().sum() or 1
            for name, v in g.sort_values(key=abs, ascending=False).items():
                drivers.append({"faktor": name, "beitrag_pct": round(v * 100, 2),
                                "gewicht_pct": round(abs(v) / tot * 100, 1),
                                "richtung": "positiv" if v > 0 else "negativ"})
            top = contrib.sort_values(key=abs, ascending=False).head(8)
            top_feats = [{"merkmal": k, "wert": None if pd.isna(x_now[k].iloc[0]) else round(float(x_now[k].iloc[0]), 4),
                          "beitrag_pct": round(v * 100, 2)} for k, v in top.items()]
        else:
            top_feats = []

        def scen(mask):
            if mask.sum() == 0:
                return {"wahrscheinlichkeit": 0.0, "preis_median": None}
            pm = float(np.median(prices[mask]))
            return {"wahrscheinlichkeit": round(float(mask.mean()) * 100, 1),
                    "preis_median": round(pm, 0), "veraenderung_pct": round((pm / p0 - 1) * 100, 1)}

        pf = p0 * np.exp(mu)
        out[h] = {
            "handelstage": n,
            "prognose_usd": round(pf, 0),
            "prognose_eur": round(pf / fx, 0) if fx else None,
            "veraenderung_pct": round((np.exp(mu) - 1) * 100, 2),
            "median_usd": round(q(0.5), 0),
            "wahrscheinlichkeit_steigend": round(p_up * 100, 1),
            "wahrscheinlichkeit_fallend": round((1 - p_up) * 100, 1),
            "szenarien": {"bullisch": scen(bull), "neutral": scen(neu), "baerisch": scen(bear),
                          "schwelle_pct": round((np.exp(thr) - 1) * 100, 1)},
            "handelsspanne_50": [round(q(0.25), 0), round(q(0.75), 0)],
            "intervall_80": [round(q(0.10), 0), round(q(0.90), 0)],
            "intervall_95": [round(q(0.025), 0), round(q(0.975), 0)],
            "volatilitaet_horizont_pct": round(sigma * 100, 2),
            "konfidenz": conf, "konfidenz_score": conf_score,
            "modelle": {k: {"veraenderung_pct": round((np.exp(v) - 1) * 100, 2), "gewicht_pct": round(w[k] * 100, 1)}
                        for k, v in preds.items()},
            "klassifikator_p_steigend": round(p_up_clf * 100, 1), "klassifikator_status": clf_note,
            "gewichtungsbasis": basis, "monte_carlo": mc_basis,
            "treiber": drivers, "top_merkmale": top_feats,
        }
    out["_garch_vol_annual_pct"] = round(garch.daily_vol_annual(), 1)
    return out
