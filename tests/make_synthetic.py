"""Erzeugt SYNTHETISCHE Testdaten im Cache (nur für Offline-Tests – nicht für echte Prognosen!)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cache = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache")
cache.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
idx = pd.bdate_range("1990-01-02", pd.Timestamp.today().normalize() - pd.offsets.BDay(1))
n = len(idx)

macro_daily = {
    "DFII10": 1.5 + np.cumsum(rng.normal(0, 0.04, n)) * 0.3, "DGS10": 3 + np.cumsum(rng.normal(0, 0.05, n)) * 0.3,
    "DGS2": 2 + np.cumsum(rng.normal(0, 0.05, n)) * 0.3, "T10YIE": 2 + np.cumsum(rng.normal(0, 0.02, n)) * 0.2,
    "T10Y2Y": 1 + np.cumsum(rng.normal(0, 0.03, n)) * 0.2, "DFF": np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)) * 0.5, 0, 6),
    "ECBDFR": np.clip(1 + np.cumsum(rng.normal(0, 0.01, n)), -0.5, 4),
    "DTWEXBGS": 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))),
    "VIXCLS": np.abs(18 + np.cumsum(rng.normal(0, 0.8, n)) * 0.1) + 10,
    "BAMLH0A0HYM2": np.abs(4 + np.cumsum(rng.normal(0, 0.05, n)) * 0.2) + 2,
    "WALCL": 4e6 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
}
# Gold reagiert (synthetisch) negativ auf Realzins- und Dollaränderungen
r = rng.standard_t(5, n) * 0.008 - 0.02 * np.diff(macro_daily["DFII10"], prepend=1.5) \
    - 0.5 * np.diff(np.log(macro_daily["DTWEXBGS"]), prepend=np.log(100)) + 0.0002
c = 400 * np.exp(np.cumsum(r))
h, l = c * (1 + np.abs(rng.normal(0, 0.004, n))), c * (1 - np.abs(rng.normal(0, 0.004, n)))
pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": h, "low": l, "close": c, "volume": 0}, index=idx).to_csv(cache / "px_gold.csv")
for name, base, vol in (("silver", 5, 0.015), ("eurusd", 1.1, 0.005), ("spx", 300, 0.011), ("gld", 100, 0.01)):
    s = base * np.exp(np.cumsum(rng.normal(0.0002, vol, n)))
    pd.DataFrame({"open": s, "high": s * 1.005, "low": s * 0.995, "close": s, "volume": 1e6}, index=idx).to_csv(cache / f"px_{name}.csv")
for k, v in macro_daily.items():
    pd.DataFrame({k: v}, index=idx).to_csv(cache / f"fred_{k}.csv")
midx = pd.date_range("1990-01-01", idx[-1] - pd.Timedelta(days=45), freq="MS")
m = len(midx)
for k, base, g in (("CPIAUCSL", 130, 0.0022), ("CPILFESL", 135, 0.002), ("PPIACO", 110, 0.002), ("PCEPI", 60, 0.0018),
                   ("PAYEMS", 110000, 0.001)):
    pd.DataFrame({k: base * np.exp(np.cumsum(rng.normal(g, 0.002, m)))}, index=midx).to_csv(cache / f"fred_{k}.csv")
pd.DataFrame({"UNRATE": np.clip(5 + np.cumsum(rng.normal(0, 0.1, m)) * 0.3, 3, 10)}, index=midx).to_csv(cache / "fred_UNRATE.csv")
pd.DataFrame({"SAHMREALTIME": np.clip(rng.normal(0.1, 0.25, m), -0.3, 1.5)}, index=midx).to_csv(cache / "fred_SAHMREALTIME.csv")
qidx = pd.date_range("1990-01-01", idx[-1] - pd.Timedelta(days=120), freq="QS")
pd.DataFrame({"GDPC1": 9000 * np.exp(np.cumsum(rng.normal(0.006, 0.005, len(qidx))))}, index=qidx).to_csv(cache / "fred_GDPC1.csv")
widx = pd.date_range("2006-01-03", idx[-1], freq="W-TUE")
net = 100000 + np.cumsum(rng.normal(0, 8000, len(widx)))
oi = 500000 + np.cumsum(rng.normal(0, 5000, len(widx)))
pd.DataFrame({"open_interest_all": oi, "noncomm_positions_long_all": net + 50000, "noncomm_positions_short_all": 50000,
              "spec_net": net, "spec_net_oi": net / oi}, index=widx.rename("date")).to_csv(cache / "cftc_cot.csv")
didx = pd.bdate_range("2005-01-03", idx[-1])
pd.DataFrame({"tonnes": 800 + np.cumsum(rng.normal(0, 2, len(didx)))}, index=didx).to_csv(cache / "etf_gld.csv")
print("SYNTHETISCHE Testdaten geschrieben nach", cache)
