"""Tagespipeline: Daten → Features → Backtest → Prognose → Bericht (JSON fürs Dashboard)."""
from __future__ import annotations

import argparse
import json
import logging
import warnings
import shutil
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import history, interpret, sentiment, sources, technical, tracking
from .backtest import direction_quality, ensemble_oos, interval_coverage, metrics, regime_metrics, walk_forward
from .config import load_config
from .features import build_features, primary_regime, regimes, targets
from .forecast import current_forecast

log = logging.getLogger("goldcast")
warnings.filterwarnings("ignore")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, date)):
        return o.isoformat()[:10]
    return o


def write_json(path, obj):
    path.write_text(json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


# ------------------------------------------------------------ Bausteine
def price_snapshot(px: dict) -> dict:
    c = px["gold"]["close"]
    fx = px["eurusd"]["close"].reindex(c.index).ffill() if "eurusd" in px else None
    ch = lambda n: round((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100, 2)
    out = {"usd": round(float(c.iloc[-1]), 2), "datum": c.index[-1].date().isoformat(),
           "datentyp": "Tagesschlusskurs (verzögert, kein Echtzeitkurs)",
           "veraenderung_1t_pct": ch(1), "veraenderung_1w_pct": ch(5), "veraenderung_1m_pct": ch(21),
           "veraenderung_ytd_pct": round((c.iloc[-1] / c[c.index.year < c.index[-1].year].iloc[-1] - 1) * 100, 2),
           "hoch_52w": round(float(px["gold"]["high"].iloc[-252:].max()), 1),
           "tief_52w": round(float(px["gold"]["low"].iloc[-252:].min()), 1)}
    if fx is not None and pd.notna(fx.iloc[-1]):
        out["eurusd"] = round(float(fx.iloc[-1]), 4)
        out["eur"] = round(float(c.iloc[-1] / fx.iloc[-1]), 2)
        out["eur_veraenderung_1m_pct"] = round(((c / fx).iloc[-1] / (c / fx).iloc[-22] - 1) * 100, 2)
    return out


def fundamentals(macro: pd.DataFrame, F: pd.DataFrame, cot, etf, cfg) -> dict:
    def snap(sid, label, unit, ch_obs):
        if sid not in macro or macro[sid].dropna().empty:
            return None
        s = macro[sid].dropna()
        v = {"bezeichnung": label, "wert": round(float(s.iloc[-1]), 3), "einheit": unit, "stand": s.index[-1].date().isoformat(),
             "quelle": f"FRED {sid}"}
        if len(s) > ch_obs:
            if unit == "Index":
                v["veraenderung_1m_pct"] = round(float((s.iloc[-1] / s.iloc[-1 - ch_obs] - 1) * 100), 2)
            else:
                v["veraenderung_1m"] = round(float(s.iloc[-1] - s.iloc[-1 - ch_obs]), 3)
        return v

    out = {
        "realrendite_10j": snap("DFII10", "US 10J Realrendite (TIPS)", "%", 21),
        "rendite_10j": snap("DGS10", "US 10J Rendite", "%", 21),
        "rendite_2j": snap("DGS2", "US 2J Rendite", "%", 21),
        "breakeven_10j": snap("T10YIE", "10J Breakeven-Inflation", "%", 21),
        "fed_funds": snap("DFF", "Fed Funds (effektiv)", "%", 21),
        "ezb_einlagensatz": snap("ECBDFR", "EZB Einlagensatz", "%", 21),
        "usd_index": snap("DTWEXBGS", "US-Dollar Broad Index (Fed; DXY selbst nicht frei verfügbar)", "Index", 21),
        "vix": snap("VIXCLS", "VIX", "Punkte", 21),
        "hy_spread": snap("BAMLH0A0HYM2", "US High-Yield-Spread", "%", 21),
        "fed_bilanz": snap("WALCL", "Fed-Bilanzsumme (QE/QT)", "Mio. USD", 4),
    }
    for k, col, lbl in (("cpi_yoy", "cpi_yoy", "CPI ggü. Vorjahr"), ("kern_cpi_yoy", "core_yoy", "Kern-CPI ggü. Vorjahr"),
                        ("pce_yoy", "pce_yoy", "PCE ggü. Vorjahr"), ("ppi_yoy", "ppi_yoy", "PPI ggü. Vorjahr")):
        if col in F and F[col].notna().any():
            out[k] = {"bezeichnung": lbl, "wert": round(float(F[col].dropna().iloc[-1]), 2), "einheit": "%",
                      "quelle": "FRED (veröffentlichungsbereinigt)"}
    for k, sid, lbl in (("arbeitslosenquote", "UNRATE", "Arbeitslosenquote"), ("sahm", "SAHMREALTIME", "Sahm-Indikator")):
        out[k] = snap(sid, lbl, "%", 1)
    # Inflationsverlauf (36 Monate) für Chart
    infl = {}
    for sid, lbl in (("CPIAUCSL", "CPI"), ("CPILFESL", "Kern-CPI"), ("PCEPI", "PCE")):
        if sid in macro:
            y = (macro[sid].dropna().pct_change(12) * 100).iloc[-36:]
            infl[lbl] = {"datum": [d.date().isoformat()[:7] for d in y.index], "wert": [round(float(v), 2) for v in y]}
    out["inflationsverlauf"] = infl
    if "DFII10" in macro and "DTWEXBGS" in macro:
        rr = macro["DFII10"].dropna().iloc[-750:]
        usd = macro["DTWEXBGS"].dropna().iloc[-750:]
        out["zinsen_dollar_verlauf"] = {"datum": [d.date().isoformat() for d in rr.index[::5]],
                                        "realrendite": [round(float(v), 2) for v in rr.iloc[::5]],
                                        "usd": [round(float(usd.asof(d)), 2) if pd.notna(usd.asof(d)) else None for d in rr.index[::5]]}
    if cot is not None and not cot.empty:
        s = cot["spec_net"].dropna()
        z = (cot["spec_net_oi"] - cot["spec_net_oi"].rolling(156).mean()) / cot["spec_net_oi"].rolling(156).std()
        out["cot"] = {"bezeichnung": "Netto-Long Spekulanten (COMEX, Kontrakte)", "wert": int(s.iloc[-1]),
                      "stand": s.index[-1].date().isoformat(), "z_score_3j": round(float(z.iloc[-1]), 2),
                      "veraenderung_4w": int(s.iloc[-1] - s.iloc[-5]) if len(s) > 5 else None, "quelle": "CFTC"}
    if etf is not None and not etf.empty:
        t = etf["tonnes"]
        out["etf_gld"] = {"bezeichnung": "SPDR GLD Goldbestand", "wert": round(float(t.iloc[-1]), 1), "einheit": "t",
                          "stand": t.index[-1].date().isoformat(),
                          "veraenderung_1m": round(float(t.iloc[-1] - t.asof(t.index[-1] - pd.Timedelta(days=30))), 1),
                          "veraenderung_3m": round(float(t.iloc[-1] - t.asof(t.index[-1] - pd.Timedelta(days=91))), 1),
                          "quelle": "spdrgoldshares.com"}
    cb = sources.manual_table(cfg, "central_banks.csv")
    out["zentralbanken"] = cb.to_dict("records") if cb is not None else \
        {"hinweis": "Keine Daten hinterlegt – Quartalswerte aus WGC 'Gold Demand Trends' in data/manual/central_banks.csv eintragen."}
    ds = sources.manual_table(cfg, "demand_supply.csv")
    out["angebot_nachfrage"] = ds.to_dict("records") if ds is not None else \
        {"hinweis": "Keine Daten hinterlegt – Quartalswerte aus WGC 'Gold Demand Trends' in data/manual/demand_supply.csv eintragen."}
    return out


def fan_chart(close: pd.Series, fc: dict, horizons: dict) -> dict:
    p0, d0 = float(close.iloc[-1]), close.index[-1]
    pts = [{"datum": d0.date().isoformat(), "p10": p0, "p25": p0, "p50": p0, "p75": p0, "p90": p0}]
    for h, n in horizons.items():
        f = fc[h]
        d = (d0 + pd.offsets.BDay(n)).date().isoformat()
        pts.append({"datum": d, "p10": f["intervall_80"][0], "p25": f["handelsspanne_50"][0], "p50": f["median_usd"],
                    "p75": f["handelsspanne_50"][1], "p90": f["intervall_80"][1], "mittel": f["prognose_usd"]})
    hist = close.iloc[-250:]
    return {"historie": {"datum": [d.date().isoformat() for d in hist.index], "preis": [round(float(v), 1) for v in hist]},
            "prognose": pts}


def load_or_run_backtest(F, Y, rets, horizons, cfg, prim, force=False):
    path = cfg["general"]["cache_dir"] / "backtest.csv.gz"
    max_age = cfg["models"]["backtest_max_age_days"]
    if path.exists() and not force and (time.time() - path.stat().st_mtime) / 86400 < max_age:
        bt = pd.read_csv(path, parse_dates=["datum"])
        bt = bt[bt["datum"].isin(F.index)]
        bt["pos"] = F.index.get_indexer(bt["datum"])
        # inzwischen realisierte Ziele nachtragen
        for h in horizons:
            m = bt["horizont"] == h
            bt.loc[m, "ist"] = Y[h].values[bt.loc[m, "pos"].values]
        bt["z"] = (bt["ist"] - bt["prognose"]) / bt["sigma"]
        log.info("Backtest aus Cache (%d Zeilen)", len(bt))
        return bt, False
    log.info("Starte Walk-Forward-Backtest …")
    t = time.time()
    raw = walk_forward(F, Y, rets, horizons, cfg, prim)
    bt = ensemble_oos(raw)
    bt.to_csv(path, index=False, compression="gzip")
    log.info("Backtest fertig in %.0f s (%d Zeilen)", time.time() - t, len(bt))
    return bt, True


# ------------------------------------------------------------ Hauptlauf
def run(force_backtest=False, offline=False):
    cfg = load_config()
    out_dir, cache = cfg["general"]["output_dir"], cfg["general"]["cache_dir"]
    meta = sources.Meta()
    horizons = cfg["horizons"]

    if offline:  # nur Cache nutzen (Tests)
        sources._get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))  # noqa: E731
    px = sources.price_series(cfg, meta)
    if "gold" not in px:
        write_json(out_dir / "latest.json", {"fehler": "Goldpreisdaten nicht verfügbar – keine Prognose erstellt.",
                                             "datenquellen": meta.items,
                                             "erstellt": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        raise SystemExit("Keine Goldpreisdaten")
    macro, lags = sources.macro_series(cfg, meta)
    cot = sources.cot_series(cfg, meta)
    etf = sources.etf_holdings(cfg, meta)
    news = sources.collect_news(cfg, meta)
    events = sources.calendar_events(cfg, meta)

    gold = px["gold"]
    for k in ("open", "high", "low"):
        if k not in gold:
            gold[k] = gold["close"]
    close = gold["close"]
    F = build_features(px, macro, lags, cot, etf)
    Y = targets(close, horizons)
    rets = np.log(close).diff()
    R = regimes(F)
    prim = R.apply(primary_regime, axis=1)
    regime_now = prim.iloc[-1]
    active = [k for k, v in R.iloc[-1].items() if v]

    bt, fresh = load_or_run_backtest(F, Y, rets, horizons, cfg, prim, force_backtest)
    met = metrics(bt, close)
    dirq = direction_quality(bt)
    fx_now = float(px["eurusd"]["close"].iloc[-1]) if "eurusd" in px else None
    fc = current_forecast(F, Y, rets, close, fx_now, horizons, cfg, regime_now, bt, met, dirq)
    garch_vol = fc.pop("_garch_vol_annual_pct")

    price = price_snapshot(px)
    tech = technical.technical_summary(gold)
    fund = fundamentals(macro, F, cot, etf, cfg)
    hist = history.historical_analysis(px, macro, F)

    senti = sentiment.analyse_news(news)
    sh_path = cfg["_root"] / "data/sentiment_history.csv"
    sh = pd.read_csv(sh_path) if sh_path.exists() else None
    senti["geopolitischer_risikoindex"] = sentiment.geo_risk_index(senti.get("geopolitik_gewicht_72h", 0), sh)
    senti["makro_risikoindex"] = sentiment.macro_risk_index(F)
    if senti.get("score") is not None:
        row = pd.DataFrame([{"datum": date.today().isoformat(), "score": senti["score"], "anzahl": senti["anzahl"],
                             "geo_gewicht": senti.get("geopolitik_gewicht_72h", 0)}])
        sh = row if sh is None else pd.concat([sh[sh["datum"] != row["datum"][0]], row])
        sh.to_csv(sh_path, index=False)

    today = date.today()
    cal = sorted([e for e in events if today <= date.fromisoformat(e["datum"]) <= today + timedelta(days=60)],
                 key=lambda e: e["datum"])
    for e in cal:
        e["erwartete_wirkung"] = interpret.EVENT_IMPACT.get(e.get("kategorie"), "Unklar – abhängig vom Ausgang.")

    log_path = cfg["_root"] / "data/forecast_log.jsonl"
    n_new = tracking.append_forecasts(log_path, price["datum"], price["usd"], fc, regime_now)
    flog = tracking.load_log(log_path)
    live = tracking.evaluate(flog, close)
    live["hash_kette_ok"] = tracking.verify_chain(flog)
    shutil.copy(log_path, out_dir / "forecast_log.jsonl")

    buy = interpret.buy_environment(fc, tech, price, cfg)
    report = {
        "erstellt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datenstand": price["datum"],
        "preis": price,
        "prognose": fc,
        "garch_volatilitaet_pa_pct": garch_vol,
        "regime": {"primaer": regime_now, "aktiv": active,
                   "verlauf_12m": prim.iloc[-252:].value_counts().to_dict()},
        "analyse_text": interpret.analysis_text(price, tech, fund, fc, senti, active, hist["analogien"]),
        "kaufumfeld": buy,
        "tagesfragen": interpret.daily_qa(price, tech, fc, senti, cal, buy, fund, active),
        "technik": tech,
        "fundamental": fund,
        "sentiment": senti,
        "kalender": cal,
        "historie": hist,
        "backtest": {
            "frisch_berechnet": fresh, "start": cfg["models"]["backtest_start"],
            "methode": "Walk-Forward, expandierendes Fenster, Neutraining alle "
                       f"{cfg['models']['refit_every']} Handelstage, Purging um den Horizont, Ensemble-Gewichte nur aus realisierten Fehlern",
            "kennzahlen": met.to_dict("records"),
            "regime_kennzahlen": regime_metrics(bt).to_dict("records"),
            "abdeckung_80_intervall": interval_coverage(bt),
            "richtung_klassifikator": dirq,
        },
        "live_auswertung": live,
        "neue_prognosen_gespeichert": n_new,
        "datenquellen": meta.items,
        "hinweise": [
            "Alle Kurse sind Tagesschlusskurse (verzögert). Keine Echtzeitdaten.",
            "Prognosen sind statistische Schätzungen mit hoher Unsicherheit; unerwartete Ereignisse (Kriege, Krisen, "
            "Zentralbankentscheidungen) können sie jederzeit entwerten.",
            "Analyse, Prognose und Kaufumfeld-Einschätzung sind getrennt; nichts davon ist eine Anlageempfehlung.",
            "FRED liefert revidierte Makrodaten; Veröffentlichungsverzüge werden berücksichtigt, Revisionen nicht.",
        ],
    }
    write_json(out_dir / "latest.json", report)
    write_json(out_dir / "chart.json", {"technik": technical.chart_payload(gold), "fan": fan_chart(close, fc, horizons)})
    log.info("Fertig: %s, Regime %s, %d neue Prognosen", price["datum"], regime_now, n_new)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-backtest", action="store_true")
    ap.add_argument("--offline", action="store_true", help="nur Cache verwenden")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.force_backtest, a.offline)


if __name__ == "__main__":
    main()
