"""Look-ahead-Test: Features eines Tages dürfen sich nicht ändern, wenn spätere Daten entfernt werden.

Ausführen (nach `python tests/make_synthetic.py` oder einem echten Lauf mit Cache):
    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldcast import sources  # noqa: E402
from goldcast.config import load_config  # noqa: E402
from goldcast.features import build_features  # noqa: E402


def _load():
    cfg = load_config()
    sources._get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    meta = sources.Meta()
    px = sources.price_series(cfg, meta)
    macro, lags = sources.macro_series(cfg, meta)
    return px, macro, lags, sources.cot_series(cfg, meta), sources.etf_holdings(cfg, meta)


def test_features_are_causal():
    px, macro, lags, cot, etf = _load()
    full = build_features(px, macro, lags, cot, etf)
    for cut in (full.index[-800], full.index[-300], full.index[-60]):
        px_c = {k: v.loc[:cut] for k, v in px.items()}
        # Makrodaten bis zum Stichtag: Werte mit Referenzdatum nach `cut` entfernen
        macro_c = macro.loc[:cut]
        cot_c = cot.loc[:cut] if cot is not None else None
        etf_c = etf.loc[:cut] if etf is not None else None
        trunc = build_features(px_c, macro_c, lags, cot_c, etf_c)
        a, b = full.loc[cut], trunc.loc[cut]
        diff = (a - b).abs()
        bad = diff[(diff > 1e-9) & a.notna() & b.notna()]
        assert bad.empty, f"Look-ahead in {list(bad.index)} am {cut.date()}"
        assert (a.isna() == b.isna()).all(), f"NaN-Muster verschieden am {cut.date()}"


if __name__ == "__main__":
    test_features_are_causal()
    print("OK – keine zukünftigen Informationen in den Features.")
