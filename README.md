# GoldCast – KI-gestütztes Goldpreis-Analyse- und Prognosesystem (MVP)

Täglich aktualisiertes, datenbasiertes System für XAU/USD: Datenabruf → Features → Walk-Forward-Backtest →
Ensemble-Prognose → Szenarien → Analystentext → Dashboard. Läuft kostenlos über **GitHub Actions** und
**GitHub Pages**, nur mit **kostenlosen Datenquellen ohne API-Key**.

> Keine Anlageberatung. Das System liefert statistische Schätzungen mit transparent ausgewiesener Unsicherheit.

## 1. Systemarchitektur

```
 GitHub Actions (Mo–Fr 22:35 UTC)
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ sources.py      Stooq/Yahoo · FRED · CFTC · SPDR GLD · GDELT · RSS · Fed · BLS │
 │      │          (Cache bei Ausfall, Metadaten: Quelle, Stand, Datentyp)       │
 │      ▼                                                                       │
 │ features.py     55+ Merkmale, Veröffentlichungsverzug je Makroreihe,         │
 │                 kein Look-ahead  ──►  regimes (9 Marktregime)                │
 │      ▼                                                                       │
 │ backtest.py     Walk-Forward (expandierend, Purging), OOS-Ensemble,          │
 │                 MAE/RMSE/MAPE/Trefferquote/Intervall-Abdeckung je Regime     │
 │      ▼                                                                       │
 │ forecast.py     Drift · Bayes-Regression · LightGBM · Random Forest · ARIMA  │
 │                 → Inverse-Fehler-Gewichtung (regimeabhängig) → GARCH-Vola    │
 │                 → Monte Carlo (Bootstrap echter OOS-Residuen) → Szenarien    │
 │      ▼                                                                       │
 │ technical.py · history.py · sentiment.py · interpret.py · tracking.py        │
 │      ▼                                                                       │
 │ docs/data/latest.json, chart.json, forecast_log.jsonl  ──► git commit        │
 └──────────────────────────────────────────────────────────────────────────────┘
                                  ▼
                 GitHub Pages: docs/index.html (Dashboard, Chart.js)
```

| Modul | Inhalt |
|---|---|
| `sources.py` | Alle Datenabrufe, Retry, Cache-Fallback, Herkunfts-Metadaten |
| `features.py` | Kausale Features, Publikationsverzüge, Faktorgruppen, Regime-Erkennung |
| `technical.py` | SMA/EMA, MACD, ADX, Ichimoku, RSI, Stochastik, CCI, Williams %R, ROC, Bollinger, ATR, Keltner, HV; S/R-Zonen, Fibonacci, Trendkanal, Marktstruktur, Candlesticks, (False) Breakouts; Tages-/Wochen-/Monatschart |
| `models.py` | Modelle (siehe unten) |
| `backtest.py` | Walk-Forward-Validierung, Kennzahlen, Regime-Auswertung, Kalibrierung |
| `forecast.py` | Aktuelle Prognose, Gewichtung, Monte Carlo, Szenarien, SHAP-Treiber, Konfidenz |
| `history.py` | Langfriststatistik, Saisonalität, Krisenreaktionen, Zinszyklen, Rezession, Korrelationen, k-NN-Analogien |
| `sentiment.py` | News-Relevanz, Deduplizierung, goldspezifisches Sentiment (5 Stufen), Glaubwürdigkeit, „bereits eingepreist“, Risikoindizes |
| `interpret.py` | Analystentext, 8 Tagesfragen, Kaufumfeld (getrennt, mit Kosten) |
| `tracking.py` | Append-only-Prognoseprotokoll mit SHA-256-Hash-Kette, Live-MAE/RMSE/MAPE/Treffer/Brier |

## 2. Datenquellen (alle kostenlos)

| Bereich | Quelle | Frequenz |
|---|---|---|
| Gold, Silber, EUR/USD, S&P 500, GLD | Stooq (Fallback Yahoo Finance) | Tagesschluss |
| Zinsen, Realrenditen, Breakeven, Fed Funds, EZB-Satz, Dollar-Index, VIX, HY-Spread, Fed-Bilanz, CPI, Kern-CPI, PPI, PCE, Arbeitsmarkt, Sahm, BIP | FRED (fredgraph.csv, kein Key) | täglich–quartalsweise |
| Futures-Positionierung | CFTC COT (Socrata-API) | wöchentlich |
| ETF-Bestände | SPDR Gold Shares (GLD, Tonnen) | täglich |
| Nachrichten | GDELT DOC API (indexiert u. a. Reuters, Bloomberg, CNBC, FT …) + RSS (Fed, EZB, Kitco, CNBC, MarketWatch, Investing, FXStreet) | laufend |
| Termine | federalreserve.gov (FOMC), bls.gov (CPI, NFP, PPI) + `data/manual/calendar.yaml` | – |
| Zentralbankkäufe, Angebot/Nachfrage | World Gold Council – **manuell**, `data/manual/*.csv` (keine freie API) | quartalsweise |

Nicht kostenlos verfügbar und daher bewusst nicht enthalten: Bloomberg-/Reuters-Terminals, DXY-Original (ersetzt durch Fed Broad Dollar Index), implizite Gold-Volatilität (GVZ ggf. über FRED `GVZCLS` ergänzbar), Intraday-Kurse.

## 3. Modelle und Validierung

- **Historischer Drift** (Benchmark) · **Bayesianische Regression** · **LightGBM** (Gradient Boosting) · **Random Forest** · **ARIMA(1,0,1)** · **GARCH(1,1)-t** für Volatilität · **LightGBM-Klassifikator** für P(steigend) – wird nur verwendet, wenn er im Backtest besser kalibriert ist als die Basisrate.
- **Ensemble**: Gewichte = inverse Out-of-Sample-MSE der letzten ~3 Jahre, bei ≥ 200 Beobachtungen zur Hälfte regimespezifisch, 50 % Schrumpfung zur Gleichgewichtung (gegen Overfitting).
- **Monte Carlo**: 5.000 Ziehungen aus den standardisierten echten OOS-Fehlern × aktuelle GARCH-Volatilität → empirisch kalibrierte Intervalle (Abdeckung wird im Dashboard ausgewiesen).
- **Szenarien**: bullisch/bärisch = Rendite über/unter ±0,5 σ des Horizonts, mit Wahrscheinlichkeit und Median-Kursziel.
- **Kein Look-ahead**: Makrodaten erst ab Veröffentlichung, Training nur auf bereits realisierten Zielen (Purging), Ensemble-Gewichte im Backtest nur aus Vergangenheit. `tests/test_no_lookahead.py` prüft das bei jedem Lauf.
- **Ehrliche Benchmarks**: Jede Kennzahl steht neben dem Drift-Modell und dem „Anteil steigend“. Schlägt ein Modell diese nicht, erhält es automatisch wenig Gewicht.

## 4. Einrichtung (ca. 10 Minuten)

1. Neues GitHub-Repository anlegen und den Inhalt dieses Ordners hochladen (`git init && git add . && git commit -m init && git push`).
2. **Settings → Actions → General → Workflow permissions**: „Read and write permissions“.
3. **Actions → „Tägliche Goldprognose“ → Run workflow** mit `force_backtest = true`. Erster Lauf ca. 15–25 Minuten (Backtest seit 2010).
4. **Settings → Pages**: Source „Deploy from a branch“, Branch `main`, Ordner `/docs`.
5. Dashboard: `https://<user>.github.io/<repo>/`. Danach läuft alles automatisch Mo–Fr; der Backtest wird wöchentlich neu berechnet.

Lokal:
```bash
pip install -r requirements.txt
python -m goldcast.pipeline --force-backtest
python -m http.server -d docs 8000   # → http://localhost:8000
```

## 5. Pflege

- Quartalsweise WGC-Zahlen in `data/manual/central_banks.csv` und `demand_supply.csv` eintragen (gold.org → Gold Demand Trends).
- EZB-Sitzungen und PCE-Termine in `data/manual/calendar.yaml`.
- Parameter (Horizonte, Kosten, Backtest-Start, Refit-Intervall) in `config.yaml`.

## 6. Grenzen (bewusst transparent)

- Tagesschlusskurse, keine Echtzeitdaten; 24-h-Prognose = nächster Handelstag.
- FRED liefert revidierte Werte (keine Echtzeit-Vintages).
- Das regelbasierte News-Sentiment fließt **nicht** in die Modelle ein, da keine ausreichende Historie existiert; es wird täglich gespeichert (`data/sentiment_history.csv`) und kann nach ~1 Jahr als Feature getestet werden.
- Goldpreisprognosen haben historisch nur geringe Richtungsgüte. Das System zeigt das offen, statt Präzision vorzutäuschen.

## 7. Ausbaustufe 2 (vorbereitet, nicht im MVP)

Intraday-Daten (15m/1h/4h, z. B. Twelve Data), LSTM/GRU/Transformer-Zeitreihenmodelle, XGBoost, SARIMA,
HMM-basierte Regime-Erkennung, FinBERT- oder LLM-Sentiment, Push-Benachrichtigung bei Regimewechsel,
GVZ-Volatilitätsindex.
