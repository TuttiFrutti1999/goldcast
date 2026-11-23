"""Prognose-Protokoll (append-only, Hash-Kette) und Live-Auswertung.

Jede Prognose wird mit Zeitstempel gespeichert. Ein Eintrag je Basisdatum und
Horizont; bestehende Einträge werden nie verändert. Die Hash-Kette macht
nachträgliche Änderungen erkennbar (`verify_chain`).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def verify_chain(log: list[dict]) -> bool:
    prev = "GENESIS"
    for r in log:
        if r.get("prev_hash") != prev or _hash(r) != r.get("hash"):
            return False
        prev = r["hash"]
    return True


def append_forecasts(path: Path, base_date: str, price: float, fc: dict, regime: str) -> int:
    log = load_log(path)
    have = {(r["basisdatum"], r["horizont"]) for r in log}
    prev = log[-1]["hash"] if log else "GENESIS"
    new = 0
    with open(path, "a", encoding="utf-8") as f:
        for h, v in fc.items():
            if h.startswith("_") or (base_date, h) in have:
                continue
            rec = {"erstellt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "basisdatum": base_date,
                   "horizont": h, "n": v["handelstage"], "preis_basis": round(price, 2),
                   "prognose": v["prognose_usd"], "p_steigend": v["wahrscheinlichkeit_steigend"],
                   "intervall_80": v["intervall_80"], "regime": regime,
                   "modelle": {k: m["veraenderung_pct"] for k, m in v["modelle"].items()}, "prev_hash": prev}
            rec["hash"] = _hash(rec)
            prev = rec["hash"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new += 1
    return new


def evaluate(log: list[dict], close: pd.Series) -> dict:
    rows = []
    idx = close.index
    for r in log:
        b = pd.Timestamp(r["basisdatum"])
        i = idx.searchsorted(b)
        if i >= len(idx) or idx[i] != b or i + r["n"] >= len(idx):
            continue
        act = float(close.iloc[i + r["n"]])
        rows.append({"horizont": r["horizont"], "basisdatum": r["basisdatum"], "zieldatum": idx[i + r["n"]].date().isoformat(),
                     "prognose": r["prognose"], "ist": round(act, 1), "basis": r["preis_basis"],
                     "p": r["p_steigend"], "lo": r["intervall_80"][0], "hi": r["intervall_80"][1],
                     **{f"m_{k}": v for k, v in r.get("modelle", {}).items()}})
    if not rows:
        return {"anzahl_ausgewertet": 0, "anzahl_gesamt": len(log), "hinweis":
                "Noch keine Prognose ist fällig. Die erste Live-Auswertung erscheint nach dem ersten Handelstag."}
    d = pd.DataFrame(rows)
    out = {"anzahl_ausgewertet": len(d), "anzahl_gesamt": len(log), "je_horizont": {}, "modelle": {}, "letzte": []}
    for h, g in d.groupby("horizont"):
        e = g["prognose"] - g["ist"]
        up = g["ist"] > g["basis"]
        out["je_horizont"][h] = {
            "n": len(g), "MAE": round(float(e.abs().mean()), 1), "RMSE": round(float(np.sqrt((e ** 2).mean())), 1),
            "MAPE": round(float((e.abs() / g["ist"]).mean() * 100), 2),
            "trefferquote_richtung": round(float(((g["prognose"] > g["basis"]) == up).mean() * 100), 1),
            "abdeckung_80": round(float(((g["ist"] >= g["lo"]) & (g["ist"] <= g["hi"])).mean() * 100), 1),
            "brier": round(float(((g["p"] / 100 - up) ** 2).mean()), 3)}
        mcols = [c for c in g.columns if c.startswith("m_")]
        for c in mcols:
            pred = g["basis"] * (1 + g[c] / 100)
            out["modelle"].setdefault(h, {})[c[2:]] = round(float(((pred - g["ist"]) ** 2).mean() ** 0.5), 1)
    out["letzte"] = d.sort_values("zieldatum", ascending=False).head(30)[
        ["horizont", "basisdatum", "zieldatum", "basis", "prognose", "ist"]].to_dict("records")
    return out
