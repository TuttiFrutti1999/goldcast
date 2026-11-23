"""Datenquellen. Alle kostenlos, ohne API-Key.

Jede Funktion liefert Daten + Metadaten (Quelle, Stand, Datentyp). Schlägt ein
Abruf fehl, wird der letzte Cache verwendet und als "Cache" gekennzeichnet –
es werden niemals Werte erfunden.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (GoldCast research tool; +https://github.com)"}
TIMEOUT = 30


def _get(url: str, params: dict | None = None, retries: int = 2) -> requests.Response:
    last = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


class Meta:
    """Sammelt Herkunftsinformationen für die Transparenz-Anzeige."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def add(self, key, source, kind, last_date=None, status="ok", note=""):
        self.items[key] = {
            "quelle": source,
            "datentyp": kind,  # "verzögert (Tagesschluss)", "historisch", "Cache", ...
            "stand": str(last_date) if last_date is not None else None,
            "status": status,
            "hinweis": note,
            "abgerufen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def _cached(cache_dir: Path, name: str, fetch, meta: Meta, source: str, kind: str):
    path = cache_dir / f"{name}.csv"
    try:
        df = fetch()
        if df is None or df.empty:
            raise ValueError("leere Antwort")
        df.to_csv(path)
        meta.add(name, source, kind, df.index.max().date() if isinstance(df.index, pd.DatetimeIndex) else None)
        return df
    except Exception as e:  # noqa: BLE001
        log.warning("Abruf %s fehlgeschlagen: %s", name, e)
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            meta.add(name, source, "Cache (Abruf fehlgeschlagen)", df.index.max().date(), "cache", str(e)[:200])
            return df
        meta.add(name, source, "nicht verfügbar", None, "fehlt", str(e)[:200])
        return None


# ---------------------------------------------------------------- Kurse
def stooq_daily(symbol: str) -> pd.DataFrame:
    r = _get("https://stooq.com/q/d/l/", {"s": symbol, "i": "d"})
    if not r.text.startswith("Date"):
        raise ValueError(f"Stooq lieferte kein CSV für {symbol}")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


YF_MAP = {"xauusd": "GC=F", "xagusd": "SI=F", "eurusd": "EURUSD=X", "^spx": "^GSPC", "gld.us": "GLD"}


def yahoo_daily(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf  # optionaler Fallback

    t = YF_MAP.get(symbol, symbol)
    df = yf.download(t, start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])


_STOOQ_DOWN = False  # nach dem ersten Fehlschlag Stooq für diesen Lauf überspringen


def price_series(cfg: dict, meta: Meta) -> dict[str, pd.DataFrame]:
    cache = cfg["general"]["cache_dir"]
    start = cfg["general"]["history_start"]
    out = {}
    for name, sym in cfg["sources"]["stooq"].items():

        def fetch(sym=sym):
            global _STOOQ_DOWN
            if not _STOOQ_DOWN:
                try:
                    r = requests.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d"}, headers=UA, timeout=10)
                    r.raise_for_status()
                    if not r.text.startswith("Date"):
                        raise ValueError("kein CSV")
                    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
                    df.columns = [c.lower() for c in df.columns]
                    return df.sort_index().loc[start:]
                except Exception as e:  # noqa: BLE001
                    _STOOQ_DOWN = True
                    log.info("Stooq nicht erreichbar (%s) – ab jetzt nur Yahoo", e)
            return yahoo_daily(sym, start)

        src = "Stooq (Fallback: Yahoo Finance)"
        if name == "gold":
            src += " – Spot XAU/USD; Yahoo-Fallback = COMEX-Future GC=F"
        df = _cached(cache, f"px_{name}", fetch, meta, src, "verzögert (Tagesschluss)")
        if df is not None:
            out[name] = df
    return out


# ---------------------------------------------------------------- FRED
def fred_series(series_id: str) -> pd.Series:
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series_id})
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series_id
    return s.dropna()


def macro_series(cfg: dict, meta: Meta) -> tuple[pd.DataFrame, dict]:
    cache = cfg["general"]["cache_dir"]
    frames, lags = {}, {}
    for sid, (desc, freq, lag) in cfg["sources"]["fred"].items():
        df = _cached(cache, f"fred_{sid}", lambda sid=sid: fred_series(sid).to_frame(), meta,
                     f"FRED – {desc}", "historisch/verzögert (Veröffentlichungsverzug %s T)" % lag)
        if df is not None:
            frames[sid] = df.iloc[:, 0]
            lags[sid] = (freq, lag)
    return pd.DataFrame(frames), lags


# ---------------------------------------------------------------- CFTC COT
def cftc_cot(code: str) -> pd.DataFrame:
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        "cftc_contract_market_code": code,
        "$limit": 5000,
        "$order": "report_date_as_yyyy_mm_dd",
        "$select": "report_date_as_yyyy_mm_dd,open_interest_all,noncomm_positions_long_all,"
                   "noncomm_positions_short_all,comm_positions_long_all,comm_positions_short_all",
    }
    rows = _get(url, params).json()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.pop("report_date_as_yyyy_mm_dd"))
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce").sort_index()
    df["spec_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["spec_net_oi"] = df["spec_net"] / df["open_interest_all"]
    return df


def cot_series(cfg: dict, meta: Meta) -> pd.DataFrame | None:
    return _cached(cfg["general"]["cache_dir"], "cftc_cot",
                   lambda: cftc_cot(cfg["sources"]["cftc"]["gold_code"]), meta,
                   "CFTC Commitments of Traders (COMEX Gold)", "wöchentlich, Stand Dienstag, Veröffentlichung Freitag")


# ---------------------------------------------------------------- News
def gdelt_news(query: str, timespan: str) -> list[dict]:
    params = {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
              "timespan": timespan, "maxrecords": 150, "sort": "datedesc"}
    data = _get("https://api.gdeltproject.org/api/v2/doc/doc", params).json()
    out = []
    for a in data.get("articles", []):
        try:
            dt = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        out.append({"titel": a.get("title", "").strip(), "url": a.get("url"), "quelle": a.get("domain"),
                    "datum": dt.isoformat(), "kanal": "GDELT"})
    return out


def rss_news(feeds: list[list[str]]) -> tuple[list[dict], dict]:
    import feedparser

    out, status = [], {}
    for name, url in feeds:
        try:
            r = _get(url, retries=1)
            fp = feedparser.parse(r.content)
            n = 0
            for e in fp.entries[:60]:
                t = e.get("published_parsed") or e.get("updated_parsed")
                if not t:
                    continue
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                out.append({"titel": re.sub(r"\s+", " ", e.get("title", "")).strip(),
                            "text": re.sub("<[^>]+>", " ", e.get("summary", ""))[:500],
                            "url": e.get("link"), "quelle": name, "datum": dt.isoformat(), "kanal": "RSS"})
                n += 1
            status[name] = f"ok ({n})"
        except Exception as e:  # noqa: BLE001
            status[name] = f"Fehler: {str(e)[:80]}"
    return out, status


def collect_news(cfg: dict, meta: Meta) -> list[dict]:
    ncfg = cfg["sources"]["news"]
    items = []
    try:
        items += gdelt_news(ncfg["gdelt_query"], ncfg["gdelt_timespan"])
        meta.add("news_gdelt", "GDELT DOC 2.0 API", "nahezu Echtzeit (15-Min-Updates)", datetime.now().date())
    except Exception as e:  # noqa: BLE001
        meta.add("news_gdelt", "GDELT DOC 2.0 API", "nicht verfügbar", None, "fehlt", str(e)[:200])
    rss, status = rss_news(ncfg["rss"])
    items += rss
    meta.add("news_rss", "RSS-Feeds", "nahezu Echtzeit", datetime.now().date(),
             "ok" if rss else "fehlt", json.dumps(status, ensure_ascii=False))
    return items


# ---------------------------------------------------------------- Kalender
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}


def fomc_dates(url: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_get(url).text, "html.parser")
    out = []
    for panel in soup.select("div.panel"):
        head = panel.find(["h4", "h5", "a"], string=re.compile(r"\d{4} FOMC Meetings"))
        if not head:
            continue
        year = int(re.search(r"(\d{4})", head.get_text()).group(1))
        for row in panel.select("div.fomc-meeting"):
            m = row.select_one(".fomc-meeting__month")
            d = row.select_one(".fomc-meeting__date")
            if not m or not d:
                continue
            month = m.get_text(strip=True).split("/")[-1]
            days = re.findall(r"\d+", d.get_text())
            if month not in MONTHS or not days:
                continue
            day = int(days[-1])  # Entscheidungstag = letzter Sitzungstag
            try:
                dt = datetime(year, MONTHS[month], day).date()
            except ValueError:
                continue
            out.append({"datum": dt.isoformat(), "ereignis": "FOMC-Zinsentscheid (Fed)",
                        "kategorie": "Zentralbank", "quelle": "federalreserve.gov"})
    return out


def bls_releases(url: str) -> list[dict]:
    text = _get(url).text.replace("\r\n ", "")
    keep = {"Consumer Price Index": ("US-Inflation (CPI)", "Inflation"),
            "Employment Situation": ("US-Arbeitsmarktbericht (NFP)", "Arbeitsmarkt"),
            "Producer Price Index": ("US-Erzeugerpreise (PPI)", "Inflation")}
    out = []
    for ev in text.split("BEGIN:VEVENT")[1:]:
        summ = re.search(r"SUMMARY:(.*)", ev)
        start = re.search(r"DTSTART[^:]*:(\d{8})", ev)
        if not summ or not start:
            continue
        for k, (name, cat) in keep.items():
            if summ.group(1).strip().startswith(k):
                out.append({"datum": datetime.strptime(start.group(1), "%Y%m%d").date().isoformat(),
                            "ereignis": name, "kategorie": cat, "quelle": "bls.gov"})
    return out


def calendar_events(cfg: dict, meta: Meta) -> list[dict]:
    c = cfg["sources"]["calendar"]
    ev = []
    for key, fn, arg in (("kal_fed", fomc_dates, c["fed_url"]), ("kal_bls", bls_releases, c["bls_ics"])):
        try:
            got = fn(arg)
            ev += got
            meta.add(key, arg, "Termine", datetime.now().date(), "ok" if got else "leer")
        except Exception as e:  # noqa: BLE001
            meta.add(key, arg, "nicht verfügbar", None, "fehlt", str(e)[:200])
    # Manuelle Ergänzungen (EZB-Sitzungen, PCE, Geopolitik)
    import yaml

    man = cfg["_root"] / "data/manual/calendar.yaml"
    if man.exists():
        extra = yaml.safe_load(man.read_text(encoding="utf-8")) or []
        for e in extra:
            e["datum"] = str(e["datum"])
            e.setdefault("quelle", "manuell gepflegt")
        ev += extra
    return ev


def manual_table(cfg: dict, name: str) -> pd.DataFrame | None:
    p = cfg["_root"] / "data/manual" / name
    if not p.exists():
        return None
    df = pd.read_csv(p, comment="#")
    return df if not df.empty else None


# ---------------------------------------------------------------- ETF-Bestände
def gld_holdings() -> pd.DataFrame:
    """Goldbestand des SPDR Gold Trust (GLD) in Tonnen – Proxy für ETF-Flüsse."""
    url = "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv"
    text = _get(url).text
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.lower().startswith("date"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    tcol = next(c for c in df.columns if "tonnes" in c.lower())
    out = pd.DataFrame({"tonnes": pd.to_numeric(df[tcol].astype(str).str.replace(",", ""), errors="coerce")})
    out.index = pd.to_datetime(df[df.columns[0]], errors="coerce", dayfirst=True)
    return out[out.index.notna()].dropna().sort_index()


def etf_holdings(cfg: dict, meta: Meta) -> pd.DataFrame | None:
    return _cached(cfg["general"]["cache_dir"], "etf_gld", gld_holdings, meta,
                   "SPDR Gold Shares (GLD) – Bestand in Tonnen", "täglich, verzögert")
