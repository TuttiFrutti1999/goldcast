"""Walk-Forward-Validierung (expandierendes Fenster, Purging gegen Look-ahead).

Zum Refit-Zeitpunkt t (Position p) wird für Horizont h nur auf Zeilen trainiert,
deren Ziel bis t bereits realisiert war (Position <= p - n). Die Prognosen für
die folgenden `refit_every` Tage nutzen nur Features dieser Tage.
Ensemble-Gewichte im Backtest werden ausschließlich aus bereits realisierten
Out-of-Sample-Fehlern berechnet.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .models import ARIMA, FEATURE_MODELS, GARCH, GBMDirection

log = logging.getLogger(__name__)


def usable_columns(X: pd.DataFrame, max_nan: float = 0.5) -> list[str]:
    return [c for c in X.columns if X[c].isna().mean() <= max_nan]


def walk_forward(F: pd.DataFrame, Y: pd.DataFrame, rets: pd.Series, horizons: dict, cfg: dict,
                 regime: pd.Series) -> pd.DataFrame:
    mcfg = cfg["models"]
    start = pd.Timestamp(mcfg["backtest_start"])
    mstart = pd.Timestamp(cfg["general"]["model_start"])
    k, seed = mcfg["refit_every"], mcfg["random_seed"]
    idx = F.index
    first = max(idx.searchsorted(start), idx.searchsorted(mstart) + mcfg["min_train"])
    recs = []
    refits = list(range(first, len(idx), k))
    for j, p in enumerate(refits):
        block = range(p, min(p + k, len(idx)))
        arima = ARIMA().fit_series(rets.iloc[: p + 1])
        garch = GARCH().fit_series(rets.iloc[: p + 1])
        for h, n in horizons.items():
            lo = idx.searchsorted(mstart)
            tr = slice(lo, p - n + 1)
            Xtr, ytr = F.iloc[tr], Y[h].iloc[tr]
            ok = ytr.notna()
            Xtr, ytr = Xtr[ok], ytr[ok]
            if len(ytr) < mcfg["min_train"] // 2:
                continue
            cols = usable_columns(Xtr)
            Xb = F.iloc[list(block)][cols]
            preds = {}
            for M in FEATURE_MODELS:
                m = M(seed=seed)
                preds[M.name] = m.fit(Xtr[cols], ytr).predict(Xb)
            preds["ARIMA"] = np.full(len(Xb), arima.forecast_cum(n))
            pup = GBMDirection(seed).fit(Xtr[cols], ytr).predict_proba(Xb)
            sig = garch.sigma_h(n)
            for i, pos in enumerate(block):
                for name, v in preds.items():
                    recs.append((idx[pos], h, n, pos, name, float(v[i]), Y[h].iloc[pos], sig, float(pup[i]),
                                 regime.iloc[pos]))
        if j % 20 == 0:
            log.info("Walk-Forward %d/%d (%s)", j + 1, len(refits), idx[p].date())
    df = pd.DataFrame(recs, columns=["datum", "horizont", "n", "pos", "modell", "prognose", "ist", "sigma", "p_up",
                                     "regime"])
    return df


def ensemble_oos(bt: pd.DataFrame, window: int = 750, shrink: float = 0.5) -> pd.DataFrame:
    """Ensemble mit Gewichten nur aus bereits realisierten Fehlern (kein Look-ahead)."""
    out = []
    for h, g in bt.groupby("horizont"):
        n = int(g["n"].iloc[0])
        wide = g.pivot_table(index="pos", columns="modell", values="prognose")
        actual = g.groupby("pos")["ist"].first()
        meta = g.groupby("pos")[["datum", "sigma", "p_up", "regime"]].first()
        models = list(wide.columns)
        err2 = (wide.sub(actual, axis=0)) ** 2
        ens = pd.Series(index=wide.index, dtype=float)
        w, last_upd = pd.Series(1 / len(models), index=models), -10**9
        for pos in wide.index:
            if pos - last_upd >= 21:  # Gewichte monatlich aktualisieren
                realized = err2.loc[(err2.index <= pos - n) & (err2.index > pos - n - window)].dropna()
                if len(realized) >= 60:
                    inv = 1 / realized.mean()
                    w = shrink * inv / inv.sum() + (1 - shrink) / len(models)
                last_upd = pos
            ens[pos] = float((wide.loc[pos] * w).sum())
        e = meta.copy()
        e["horizont"], e["n"], e["modell"], e["prognose"], e["ist"] = h, n, "Ensemble", ens, actual
        out.append(e.reset_index())
    ens_df = pd.concat(out, ignore_index=True)
    res = pd.concat([bt, ens_df[bt.columns]], ignore_index=True)
    # standardisierte Residuen für Kalibrierung
    res["z"] = (res["ist"] - res["prognose"]) / res["sigma"]
    return res


def interval_coverage(ens: pd.DataFrame, q=(0.1, 0.9)) -> dict:
    """Trefferquote der 80%-Intervalle, Quantile nur aus Vergangenheit."""
    out = {}
    for h, g in ens[ens["modell"] == "Ensemble"].dropna(subset=["ist"]).groupby("horizont"):
        n = int(g["n"].iloc[0])
        g = g.sort_values("pos")
        hits, tot = 0, 0
        zs = g.set_index("pos")["z"]
        for pos, row in g.set_index("pos").iloc[::5].iterrows():
            past = zs[zs.index <= pos - n].dropna()
            if len(past) < 120:
                continue
            lo, hi = past.quantile(q[0]), past.quantile(q[1])
            tot += 1
            hits += int(lo <= row["z"] <= hi)
        out[h] = round(hits / tot * 100, 1) if tot else None
    return out


def metrics(bt: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """MAE/RMSE/MAPE auf Preisebene + Richtungstrefferquote."""
    d = bt.dropna(subset=["ist"]).copy()
    p0 = close.iloc[d["pos"].values].values
    d["preis_prog"] = p0 * np.exp(d["prognose"])
    d["preis_ist"] = p0 * np.exp(d["ist"])
    d["err"] = d["preis_prog"] - d["preis_ist"]
    d["hit"] = np.sign(d["prognose"]) == np.sign(d["ist"])
    rows = []
    for (h, m), g in d.groupby(["horizont", "modell"]):
        rows.append({"horizont": h, "modell": m, "n_obs": len(g),
                     "MAE": round(g["err"].abs().mean(), 1),
                     "RMSE": round(np.sqrt((g["err"] ** 2).mean()), 1),
                     "MAPE": round((g["err"].abs() / g["preis_ist"]).mean() * 100, 2),
                     "RMSE_rendite_pct": round(np.sqrt(((g["prognose"] - g["ist"]) ** 2).mean()) * 100, 2),
                     "trefferquote_richtung": round(g["hit"].mean() * 100, 1),
                     "anteil_steigend": round((g["ist"] > 0).mean() * 100, 1)})
    return pd.DataFrame(rows)


def regime_metrics(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt.dropna(subset=["ist"])
    rows = []
    for (h, r, m), g in d.groupby(["horizont", "regime", "modell"]):
        if len(g) < 60:
            continue
        rows.append({"horizont": h, "regime": r, "modell": m, "n_obs": len(g),
                     "RMSE_rendite_pct": round(np.sqrt(((g["prognose"] - g["ist"]) ** 2).mean()) * 100, 2),
                     "trefferquote_richtung": round((np.sign(g["prognose"]) == np.sign(g["ist"])).mean() * 100, 1)})
    return pd.DataFrame(rows)


def direction_quality(bt: pd.DataFrame) -> dict:
    d = bt[bt["modell"] == "Ensemble"].dropna(subset=["ist"])
    out = {}
    for h, g in d.groupby("horizont"):
        up = (g["ist"] > 0).astype(int)
        out[h] = {"brier": round(float(((g["p_up"] - up) ** 2).mean()), 3),
                  "brier_basis": round(float(((up.mean() - up) ** 2).mean()), 3),
                  "treffer_klassifikator": round(float(((g["p_up"] > 0.5) == up.astype(bool)).mean() * 100), 1)}
    return out
