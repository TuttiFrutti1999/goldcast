/* GoldCast Dashboard – liest ausschließlich die von der Pipeline erzeugten JSON-Dateien. */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf = (x, d = 0) => (x == null || isNaN(x) ? "–" : Number(x).toLocaleString("de-DE", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (x, d = 1) => (x == null ? "–" : `<span class="${x >= 0 ? "up" : "dn"}">${x >= 0 ? "+" : ""}${nf(x, d)} %</span>`);
const kpi = (l, v, s = "") => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
const table = (cols, rows) =>
  `<table><thead><tr>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${cols.map((c) => `<td>${c[2] ? c[2](r[c[0]], r) : esc(r[c[0]] ?? "–")}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
const H_LABEL = { "1d": "24 Std.", "7d": "7 Tage", "30d": "30 Tage", "90d": "3 Monate", "180d": "6 Monate", "365d": "12 Monate" };
const GOLD = "#d9a93f", MUT = "#8b92a1", GRID = "#262a33";
Chart.defaults.color = MUT;
Chart.defaults.borderColor = GRID;
Chart.defaults.font.family = "system-ui";
Chart.defaults.scales.category.ticks.maxTicksLimit = 8;
Chart.defaults.scales.category.ticks.maxRotation = 0;
const charts = {};
const mk = (id, cfg) => { charts[id]?.destroy(); charts[id] = new Chart($(id), cfg); };
const baseOpts = (extra = {}) => ({ responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
  plugins: { legend: { labels: { boxWidth: 10 } } }, elements: { point: { radius: 0 } }, ...extra });

document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button, .tab").forEach((e) => e.classList.remove("on"));
    b.classList.add("on");
    $(b.dataset.t).classList.add("on");
  }));

async function load() {
  try {
    const [d, c] = await Promise.all([fetch("data/latest.json", { cache: "no-store" }).then((r) => r.json()),
      fetch("data/chart.json", { cache: "no-store" }).then((r) => r.json()).catch(() => null)]);
    if (d.fehler) throw new Error(d.fehler);
    render(d, c);
  } catch (e) {
    $("err").hidden = false;
    $("err").innerHTML = `<b>Keine Daten.</b> ${esc(e.message)}<br><span class="muted">Die Pipeline muss mindestens einmal gelaufen sein (GitHub Action „Tägliche Goldprognose“).</span>`;
    $("stamp").textContent = "keine Daten";
  }
}

function render(d, c) {
  const p = d.preis, fc = d.prognose;
  $("stamp").innerHTML = `Datenstand ${esc(d.datenstand)} · erstellt ${new Date(d.erstellt).toLocaleString("de-DE")}`;

  // ---------- Übersicht
  $("kpis").innerHTML =
    kpi("XAU/USD", nf(p.usd, 0) + " $", `${esc(p.datentyp)}`) +
    kpi("Gold in EUR", p.eur ? nf(p.eur, 0) + " €" : "–", p.eurusd ? `EUR/USD ${nf(p.eurusd, 4)}` : "") +
    kpi("1 Tag", pct(p.veraenderung_1t_pct, 2)) + kpi("1 Woche", pct(p.veraenderung_1w_pct)) +
    kpi("1 Monat", pct(p.veraenderung_1m_pct)) + kpi("Seit Jahresbeginn", pct(p.veraenderung_ytd_pct)) +
    kpi("52W-Hoch / -Tief", `${nf(p.hoch_52w)} / ${nf(p.tief_52w)}`, "USD") +
    kpi("P(steigend) 30 T.", fc["30d"] ? nf(fc["30d"].wahrscheinlichkeit_steigend) + " %" : "–", `Konfidenz: ${esc(fc["30d"]?.konfidenz)}`);
  $("analysis").innerHTML = d.analyse_text.map((t) => `<p>${esc(t)}</p>`).join("");
  $("regime").innerHTML = `<span class="chip">Regime: <b>${esc(d.regime.primaer)}</b></span>` +
    d.regime.aktiv.filter((r) => r !== d.regime.primaer).map((r) => `<span class="chip">${esc(r)}</span>`).join("");
  const b = d.kaufumfeld;
  $("buyBadge").textContent = b.einschaetzung;
  $("buyBadge").className = "badge " + (b.einschaetzung.includes("günstig") && !b.einschaetzung.includes("un") ? "g" : b.einschaetzung.includes("ungünstig") ? "r" : "");
  $("buy").innerHTML = `<ul>${(b.gruende || []).map((g) => `<li>${esc(g)}</li>`).join("")}</ul><p class="muted">${esc(b.kosten_hinweis)}</p><p class="muted"><b>${esc(b.disclaimer)}</b></p>`;
  $("qa").innerHTML = `<dl class="qa">${d.tagesfragen.map((q, i) => `<dt>${i + 1}. ${esc(q.frage)}</dt><dd>${esc(q.antwort)}</dd>`).join("")}</dl>`;

  // ---------- Prognose
  const hs = Object.keys(fc);
  $("fcTable").innerHTML = table(
    [["h", "Horizont", (v) => H_LABEL[v] || v], ["usd", "Prognose USD", (v) => nf(v)], ["eur", "EUR", (v) => nf(v)],
      ["chg", "Veränd.", (v) => pct(v)], ["pu", "P(↑)", (v) => nf(v) + " %"], ["r80", "80 %-Spanne", (v) => v],
      ["bull", "Bullisch", (v) => v], ["neu", "Neutral", (v) => v], ["bear", "Bärisch", (v) => v], ["k", "Konfidenz", (v) => esc(v)]],
    hs.map((h) => { const f = fc[h], s = f.szenarien, sc = (x) => `${nf(x.wahrscheinlichkeit)} % · ${nf(x.preis_median)}`;
      return { h, usd: f.prognose_usd, eur: f.prognose_eur, chg: f.veraenderung_pct, pu: f.wahrscheinlichkeit_steigend,
        r80: `${nf(f.intervall_80[0])}–${nf(f.intervall_80[1])}`, bull: sc(s.bullisch), neu: sc(s.neutral), bear: sc(s.baerisch), k: f.konfidenz }; }));
  $("hSel").innerHTML = hs.map((h) => `<option value="${h}" ${h === "30d" ? "selected" : ""}>${H_LABEL[h] || h}</option>`).join("");
  const det = () => {
    const f = fc[$("hSel").value], s = f.szenarien;
    const sc = (n, x, cls) => `<div><div class="muted">${n}</div><div class="v ${cls}" style="font-size:20px;font-weight:700">${nf(x.wahrscheinlichkeit)} %</div><div>Median ${nf(x.preis_median)} $ (${pct(x.veraenderung_pct)})</div></div>`;
    $("fcDetail").innerHTML =
      `<div class="scen">${sc("Bullisch (> +" + nf(s.schwelle_pct, 1) + " %)", s.bullisch, "up")}${sc("Neutral", s.neutral, "")}${sc("Bärisch (< −" + nf(s.schwelle_pct, 1) + " %)", s.baerisch, "dn")}</div>` +
      `<p>Erwartete Handelsspanne (50 %): <b>${nf(f.handelsspanne_50[0])}–${nf(f.handelsspanne_50[1])} $</b> · 95 %-Intervall: ${nf(f.intervall_95[0])}–${nf(f.intervall_95[1])} $ · Volatilität im Horizont: ${nf(f.volatilitaet_horizont_pct, 1)} %</p>` +
      `<p class="muted">Konfidenz ${esc(f.konfidenz)} (${f.konfidenz_score}/100) · Gewichtung: ${esc(f.gewichtungsbasis)} · Monte Carlo: ${esc(f.monte_carlo)} · Richtungs-Klassifikator: ${nf(f.klassifikator_p_steigend)} % – ${esc(f.klassifikator_status)}</p>` +
      `<h2 style="margin-top:14px">Modelle & Gewichte</h2>` +
      table([["m", "Modell"], ["c", "Erw. Veränd.", (v) => pct(v, 2)], ["w", "Gewicht", (v) => `${nf(v, 1)} %<div class="bar"><i style="width:${v}%"></i></div>`]],
        Object.entries(f.modelle).map(([m, v]) => ({ m, c: v.veraenderung_pct, w: v.gewicht_pct }))) +
      `<h2 style="margin-top:14px">Einflussfaktoren (LightGBM-SHAP)</h2>` +
      table([["faktor", "Faktorgruppe"], ["beitrag_pct", "Beitrag (Pp.)", (v) => pct(v, 2)], ["gewicht_pct", "Gewicht", (v) => `${nf(v, 1)} %<div class="bar"><i style="width:${v}%"></i></div>`]], f.treiber) +
      `<details><summary class="muted">Einzelmerkmale</summary>${table([["merkmal", "Merkmal"], ["wert", "Wert", (v) => nf(v, 3)], ["beitrag_pct", "Beitrag", (v) => pct(v, 2)]], f.top_merkmale)}</details>`;
  };
  $("hSel").onchange = det; det();
  if (c?.fan) {
    // Zeitachse: Historie + alle Handelstage bis zum längsten Horizont, Prognosepunkte an korrekter Position
    const H = c.fan.historie, P = c.fan.prognose, n0 = H.datum.length - 1;
    const maxN = Math.max(...Object.values(fc).map((f) => f.handelstage));
    const labels = [...H.datum];
    let dt = new Date(H.datum[n0]);
    for (let i = 0; i < maxN; i++) { do { dt.setDate(dt.getDate() + 1); } while ([0, 6].includes(dt.getDay())); labels.push(dt.toISOString().slice(0, 10)); }
    const steps = [0, ...Object.values(fc).map((f) => f.handelstage)];
    const series = (k) => { const a = Array(labels.length).fill(null); P.forEach((x, i) => (a[n0 + steps[i]] = x[k])); return a; };
    const ds = (label, k, o) => ({ label, data: series(k), spanGaps: true, borderColor: "transparent", tension: 0.3, ...o });
    mk("fanChart", { type: "line", data: { labels, datasets: [
      { label: "Goldpreis", data: [...H.preis, ...Array(maxN).fill(null)], borderColor: GOLD, borderWidth: 2 },
      ds("90 %", "p90", { backgroundColor: "rgba(217,169,63,.12)", fill: "+3" }),
      ds("75 %", "p75", { backgroundColor: "rgba(217,169,63,.25)", fill: "+1" }),
      ds("Median", "p50", { borderColor: "#fff", borderDash: [5, 4], borderWidth: 1.5, pointRadius: 3 }),
      ds("25 %", "p25", {}), ds("10 %", "p10", {})] },
      options: baseOpts({ scales: { x: { ticks: { maxTicksLimit: 10 } } }, plugins: { legend: { labels: { filter: (i) => ["Goldpreis", "Median"].includes(i.text) } } } }) });
  }

  // ---------- Sentiment
  const s = d.sentiment;
  const meter = (v) => v == null ? "–" : `${v > 0 ? "+" : ""}${nf(v)}<div class="meter"><b style="left:calc(${(v + 100) / 2}% - 2px)"></b></div>`;
  $("sentKpis").innerHTML = kpi("Nachrichten-Score (−100…+100)", meter(s.score), `${s.anzahl} relevante Meldungen (dedupliziert)`) +
    kpi("Geopolitischer Risikoindex", s.geopolitischer_risikoindex?.wert ?? "–", esc(s.geopolitischer_risikoindex?.hinweis) + ` · ${s.geopolitik_meldungen_72h ?? 0} Meldungen/72 h`) +
    kpi("Makro-Risikoindex (0–100)", s.makro_risikoindex?.wert ?? "–", Object.entries(s.makro_risikoindex?.komponenten || {}).map(([k, v]) => `${esc(k)}: ${v}`).join(" · "));
  const news = (L) => (L || []).map((n) => `<div class="news"><a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.titel)}</a>
    <div class="m">${esc(n.quelle)} · ${new Date(n.datum).toLocaleString("de-DE")} · <b class="${n.score > 0 ? "up" : n.score < 0 ? "dn" : ""}">${esc(n.einstufung)}</b> · Wirkung ${esc(n.marktwirkung)}${n.bereits_eingepreist ? " · vermutlich eingepreist" : ""} · ${esc((n.kategorien || []).join(", "))}</div></div>`).join("") || `<p class="muted">${esc(s.hinweis || "Keine Meldungen.")}</p>`;
  $("newsPos").innerHTML = news(s.top_positiv); $("newsNeg").innerHTML = news(s.top_negativ); $("newsAll").innerHTML = news(s.nachrichten);

  // ---------- Technik
  const t = d.technik, T = c?.technik;
  if (T) {
    const L = T.datum.slice(-500), sl = (k) => T[k].slice(-500);
    mk("pxChart", { type: "line", data: { labels: L, datasets: [
      { label: "Kurs", data: sl("close"), borderColor: GOLD, borderWidth: 1.8 },
      { label: "SMA 20", data: sl("sma20"), borderColor: "#6ea8fe", borderWidth: 1 },
      { label: "SMA 50", data: sl("sma50"), borderColor: "#b388ff", borderWidth: 1 },
      { label: "SMA 200", data: sl("sma200"), borderColor: "#fff", borderWidth: 1 },
      { label: "BB oben", data: sl("bb_up"), borderColor: "rgba(139,146,161,.5)", borderWidth: 1, borderDash: [3, 3] },
      { label: "BB unten", data: sl("bb_low"), borderColor: "rgba(139,146,161,.5)", borderWidth: 1, borderDash: [3, 3] }] }, options: baseOpts() });
    mk("rsiChart", { type: "line", data: { labels: L.slice(-250), datasets: [{ label: "RSI", data: sl("rsi14").slice(-250), borderColor: GOLD, borderWidth: 1.5 }] },
      options: baseOpts({ scales: { y: { min: 0, max: 100, ticks: { stepSize: 30 } } }, plugins: { legend: { display: false } } }) });
    mk("macdChart", { type: "bar", data: { labels: L.slice(-250), datasets: [
      { type: "line", label: "MACD", data: sl("macd").slice(-250), borderColor: GOLD, borderWidth: 1.5 },
      { type: "line", label: "Signal", data: sl("macd_signal").slice(-250), borderColor: "#6ea8fe", borderWidth: 1 },
      { label: "Histogramm", data: sl("macd_hist").slice(-250), backgroundColor: sl("macd_hist").slice(-250).map((v) => (v >= 0 ? "rgba(63,178,127,.6)" : "rgba(224,96,94,.6)")) }] }, options: baseOpts() });
  }
  $("tfs").innerHTML = table([["tf", "Zeitebene"], ["trend", "Trend"], ["rsi", "RSI", (v) => nf(v, 1)], ["momentum", "Momentum"], ["adx", "ADX", (v) => nf(v, 1)]],
    Object.entries(t.zeitebenen).map(([tf, v]) => ({ tf, ...v }))) + `<p class="muted">${esc(t.hinweis_intraday)}</p>`;
  $("sr").innerHTML = table([["preis", "Widerstand", (v) => nf(v)], ["beruehrungen", "Berührungen"]], t.widerstaende) +
    table([["preis", "Unterstützung", (v) => nf(v)], ["beruehrungen", "Berührungen"]], t.unterstuetzungen);
  const I = t.indikatoren;
  $("ind").innerHTML = table([["k", "Indikator"], ["v", "Wert", (v) => nf(v, 2)]], Object.entries(I).map(([k, v]) => ({ k: k.toUpperCase(), v }))) +
    `<p class="muted">Ichimoku: Kurs ${esc(t.ichimoku)}</p>`;
  const fib = t.fibonacci, tc = t.trendkanal;
  $("struct").innerHTML = `<p><b>Marktstruktur:</b> ${esc(t.marktstruktur)}</p>
    <p><b>Trendkanal (100 T.):</b> ${nf(tc.unten)} – ${nf(tc.mitte)} – ${nf(tc.oben)} $, Steigung ${pct(tc.steigung_pct_pro_jahr)} p.a., R² ${nf(tc.r2, 2)}</p>
    <p><b>Fibonacci (52 W., ${esc(fib.richtung)}):</b> ${Object.entries(fib.retracements).map(([k, v]) => `${k}: ${nf(v)}`).join(" · ")}<br>Extensions: ${Object.entries(fib.extensions).map(([k, v]) => `${k}: ${nf(v)}`).join(" · ")}</p>
    <p><b>Candlesticks (5 T.):</b> ${t.candlesticks.map((x) => `${esc(x.datum)} ${esc(x.muster)}`).join("; ") || "keine"}</p>
    <p><b>Breakouts (10 T.):</b> ${t.breakouts.map((x) => `${esc(x.datum)} ${esc(x.typ)} (${esc(x.status)})`).join("; ") || "keine"}</p>`;

  // ---------- Fundamental
  const f = d.fundamental;
  const fk = (o, unit = "") => o ? kpi(esc(o.bezeichnung), nf(o.wert, 2) + (o.einheit === "%" ? " %" : ""),
    (o.veraenderung_1m != null ? `1M: ${o.veraenderung_1m > 0 ? "+" : ""}${nf(o.veraenderung_1m, 2)}` : o.veraenderung_1m_pct != null ? `1M: ${pct(o.veraenderung_1m_pct)}` : "") + (o.stand ? ` · Stand ${esc(o.stand)}` : "")) : "";
  $("fundKpis").innerHTML = ["realrendite_10j", "rendite_10j", "rendite_2j", "fed_funds", "ezb_einlagensatz", "usd_index", "breakeven_10j",
    "cpi_yoy", "kern_cpi_yoy", "pce_yoy", "arbeitslosenquote", "sahm", "vix", "hy_spread"].map((k) => fk(f[k])).join("");
  if (f.zinsen_dollar_verlauf) {
    const z = f.zinsen_dollar_verlauf;
    mk("rrChart", { type: "line", data: { labels: z.datum, datasets: [
      { label: "10J-Realrendite %", data: z.realrendite, borderColor: GOLD, yAxisID: "y" },
      { label: "USD-Index", data: z.usd, borderColor: "#6ea8fe", yAxisID: "y1" }] },
      options: baseOpts({ scales: { y: { position: "left" }, y1: { position: "right", grid: { display: false } } } }) });
  }
  const iv = f.inflationsverlauf || {}, cols = [GOLD, "#6ea8fe", "#b388ff"];
  const ik = Object.keys(iv);
  if (ik.length) mk("inflChart", { type: "line", data: { labels: iv[ik[0]].datum, datasets: ik.map((k, i) => ({ label: k, data: iv[k].wert, borderColor: cols[i] })) }, options: baseOpts() });
  $("pos").innerHTML = (f.cot ? `<p><b>${esc(f.cot.bezeichnung)}:</b> ${nf(f.cot.wert)} (Δ 4 W.: ${nf(f.cot.veraenderung_4w)}), z-Score 3 J.: ${nf(f.cot.z_score_3j, 2)} · Stand ${esc(f.cot.stand)}</p>` : "<p class='muted'>COT nicht verfügbar</p>") +
    (f.etf_gld ? `<p><b>${esc(f.etf_gld.bezeichnung)}:</b> ${nf(f.etf_gld.wert, 1)} t · 1M: ${nf(f.etf_gld.veraenderung_1m, 1)} t · 3M: ${nf(f.etf_gld.veraenderung_3m, 1)} t · Stand ${esc(f.etf_gld.stand)}</p>` : "<p class='muted'>ETF-Bestände nicht verfügbar</p>") +
    (f.fed_bilanz ? `<p><b>Fed-Bilanz:</b> ${nf(f.fed_bilanz.wert / 1e6, 2)} Bio. USD (Δ 4 W.: ${nf((f.fed_bilanz.veraenderung_1m || 0) / 1e3, 0)} Mrd.)</p>` : "");
  const man = (x, title) => Array.isArray(x) ? `<h2>${title}</h2>` + table(Object.keys(x[0]).map((k) => [k, k]), x) : `<p class="muted"><b>${title}:</b> ${esc(x.hinweis)}</p>`;
  $("cb").innerHTML = man(f.zentralbanken, "Zentralbankkäufe") + man(f.angebot_nachfrage, "Angebot & Nachfrage");

  // ---------- Kalender
  $("cal").innerHTML = d.kalender.length ? table([["datum", "Datum"], ["ereignis", "Ereignis"], ["kategorie", "Kategorie"], ["erwartete_wirkung", "Erwartete Wirkung auf Gold"], ["quelle", "Quelle"]], d.kalender)
    : "<p class='muted'>Keine Termine geladen. EZB-/PCE-Termine können in data/manual/calendar.yaml ergänzt werden.</p>";

  // ---------- Historie
  const h = d.historie, st = h.statistik;
  $("histKpis").innerHTML = kpi("Allzeithoch (Zeitraum)", nf(st.allzeithoch.preis) + " $", esc(st.allzeithoch.datum)) +
    kpi("Rendite p.a.", pct(st.rendite_pa_pct, 2), `${esc(st.zeitraum[0])} – ${esc(st.zeitraum[1])}`) +
    kpi("Volatilität p.a.", nf(st.volatilitaet_pa_pct, 1) + " %", `GARCH aktuell: ${nf(d.garch_volatilitaet_pa_pct, 1)} %`) +
    kpi("Max. Drawdown", pct(st.max_drawdown_pct), esc(st.max_drawdown_datum)) + kpi("Aktueller Drawdown", pct(st.aktueller_drawdown_pct));
  mk("seasChart", { type: "bar", data: { labels: h.saisonalitaet.map((x) => x.monat), datasets: [{ label: "Ø Monatsrendite %", data: h.saisonalitaet.map((x) => x.mittel_pct),
    backgroundColor: h.saisonalitaet.map((x) => (x.mittel_pct >= 0 ? "rgba(63,178,127,.7)" : "rgba(224,96,94,.7)")) }] }, options: baseOpts({ plugins: { legend: { display: false } } }) });
  $("corr").innerHTML = table([["anlage", "Anlage"], ["1J", "1 J.", (v) => nf(v, 2)], ["3J", "3 J.", (v) => nf(v, 2)], ["10J", "10 J.", (v) => nf(v, 2)]], h.korrelationen);
  $("crises").innerHTML = table([["datum", "Datum"], ["ereignis", "Ereignis"], ["1M_pct", "1M", (v) => pct(v)], ["3M_pct", "3M", (v) => pct(v)], ["6M_pct", "6M", (v) => pct(v)], ["12M_pct", "12M", (v) => pct(v)]], h.krisen);
  $("cycles").innerHTML = table([["datum", "Signal"], ["ereignis", "Ereignis"], ["3M_pct", "3M", (v) => pct(v)], ["6M_pct", "6M", (v) => pct(v)], ["12M_pct", "12M", (v) => pct(v)]], h.zinszyklen) +
    (h.rezession ? `<p class="muted">Ø Monatsrendite bei Rezessionssignal (Sahm ≥ 0,5): ${pct(h.rezession.mittel_monat_rezessionssignal_pct, 2)} vs. sonst ${pct(h.rezession.mittel_monat_sonst_pct, 2)} (${h.rezession.monate_signal} Monate)</p>` : "");
  const a = h.analogien;
  $("analogs").innerHTML = a.episoden ? `<p>Median danach (3 M.): ${pct(a.median_3M_pct)} · steigend in ${nf(a.anteil_steigend)} % der Fälle · Merkmale: ${esc(a.merkmale.join(", "))}</p>` +
    table([["datum", "Datum"], ["aehnlichkeit", "Ähnlichkeit", (v) => nf(v, 2)], ["goldpreis_danach_3M_pct", "Gold danach 3M", (v) => pct(v)]], a.episoden) : `<p class="muted">${esc(a.hinweis)}</p>`;

  // ---------- Prognosegüte
  const lv = d.live_auswertung;
  $("live").innerHTML = `<p class="muted">${lv.anzahl_gesamt} gespeicherte Prognosen, ${lv.anzahl_ausgewertet} fällig/ausgewertet · Hash-Kette ${lv.hash_kette_ok ? "✓ unverändert" : "⚠ verletzt"}</p>` +
    (lv.je_horizont ? table([["h", "Horizont", (v) => H_LABEL[v] || v], ["n", "n"], ["MAE", "MAE $", (v) => nf(v, 1)], ["RMSE", "RMSE $", (v) => nf(v, 1)], ["MAPE", "MAPE", (v) => nf(v, 2) + " %"],
      ["trefferquote_richtung", "Richtung", (v) => nf(v, 1) + " %"], ["abdeckung_80", "80 %-Abdeckung", (v) => nf(v, 1) + " %"], ["brier", "Brier", (v) => nf(v, 3)]],
      Object.entries(lv.je_horizont).map(([h, v]) => ({ h, ...v }))) : `<p class="muted">${esc(lv.hinweis)}</p>`);
  const bt = d.backtest;
  $("btInfo").innerHTML = `${esc(bt.methode)}. Ab ${esc(bt.start)}. 80 %-Intervall-Abdeckung (Soll 80 %): ${Object.keys(H_LABEL).filter((h) => h in bt.abdeckung_80_intervall).map((h) => [h, bt.abdeckung_80_intervall[h]]).map(([h, v]) => `${H_LABEL[h] || h} ${nf(v, 1)} %`).join(" · ")}. „Anteil steigend“ = Trefferquote eines Modells, das immer „steigt“ sagt.`;
  $("bt").innerHTML = table([["horizont", "Horizont", (v) => H_LABEL[v] || v], ["modell", "Modell"], ["n_obs", "n"], ["MAE", "MAE $", (v) => nf(v, 1)], ["RMSE", "RMSE $", (v) => nf(v, 1)],
    ["MAPE", "MAPE", (v) => nf(v, 2) + " %"], ["trefferquote_richtung", "Richtung", (v) => nf(v, 1) + " %"], ["anteil_steigend", "Anteil steigend", (v) => nf(v, 1) + " %"]],
    bt.kennzahlen.sort((x, y) => Object.keys(H_LABEL).indexOf(x.horizont) - Object.keys(H_LABEL).indexOf(y.horizont) || x.RMSE - y.RMSE));
  const best = {};
  bt.regime_kennzahlen.forEach((r) => { const k = r.horizont + "|" + r.regime; if (!best[k] || r.RMSE_rendite_pct < best[k].RMSE_rendite_pct) best[k] = r; });
  $("btReg").innerHTML = table([["horizont", "Horizont", (v) => H_LABEL[v] || v], ["regime", "Regime"], ["modell", "Bestes Modell"], ["RMSE_rendite_pct", "RMSE (Rendite, Pp.)", (v) => nf(v, 2)], ["trefferquote_richtung", "Richtung", (v) => nf(v, 1) + " %"], ["n_obs", "n"]],
    Object.values(best).filter((r) => ["30d", "90d"].includes(r.horizont)));

  // ---------- Quellen
  $("src").innerHTML = table([["k", "Datensatz"], ["quelle", "Quelle"], ["datentyp", "Datentyp"], ["stand", "Stand"], ["status", "Status", (v) => `<span class="${v === "ok" ? "up" : "dn"}">${esc(v)}</span>`]],
    Object.entries(d.datenquellen).map(([k, v]) => ({ k, ...v })));
  $("notes").innerHTML = d.hinweise.map((n) => `<li>${esc(n)}</li>`).join("");
}
load();
