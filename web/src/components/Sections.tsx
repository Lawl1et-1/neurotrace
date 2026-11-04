import { useState } from "react";
import type { Index, SummaryRow } from "../data";

const LABELS: Record<string, string> = {
  full: "NEUROTRACE (learned graph + GAT + Transformer)",
  diffusion_gnn: "diffusion GNN instead of GAT",
  static_graph: "distance-prior graph only",
  no_graph: "no graph (channels mixed only at readout)",
  no_transformer: "no Transformer",
  no_var_evidence: "edge module without VAR evidence",
  masked_objective: "masked-window objective",
  supervised_head: "+ supervised head (uses labels)",
  channel_cnn_lstm: "Per-channel CNN-LSTM forecaster",
  baseline_bandpower_dev: "Band-power deviation",
  baseline_pearson: "Pearson deviation",
  baseline_coherence: "Coherence deviation",
  baseline_granger: "Granger (per window)",
  baseline_pdc: "PDC (per window)",
  baseline_dtf: "DTF (per window)",
};
const ORDER = Object.keys(LABELS);
/** Ablations of the full model are indented under it. */
const isAblation = (v: string) => !["full", "channel_cnn_lstm"].includes(v) && !v.startsWith("baseline_");

const num = (r: SummaryRow, k: string) => (typeof r[`${k}_mean`] === "number" ? (r[`${k}_mean`] as number) : null);
const ci = (r: SummaryRow, k: string) => (typeof r[`${k}_ci95`] === "number" ? (r[`${k}_ci95`] as number) : null);
const nOf = (r: SummaryRow, k: string) => (typeof r[`${k}_n`] === "number" ? (r[`${k}_n`] as number) : null);
/** "0.84 ± 0.05": mean and half-width of the 95 % CI over test subjects. */
const fci = (r: SummaryRow, k: string, d = 2) => {
  const m = num(r, k);
  if (m == null || Number.isNaN(m)) return "—";
  const c = ci(r, k);
  return c == null || Number.isNaN(c) ? m.toFixed(d) : `${m.toFixed(d)} ± ${c.toFixed(d)}`;
};

function Page({ id, code, time, children }: { id: string; code: string; time: string; children: React.ReactNode }) {
  return (
    <section className="page ruled" id={id}>
      <div className="page-gutter">
        <span className="code">{code}</span>
        <span className="time tnum">{time}</span>
      </div>
      <div className="page-body">{children}</div>
    </section>
  );
}

const PIPELINE: { code: string; title: string; text: string; ann: string }[] = [
  { code: "W", title: "Windows", ann: "2 s · 1 s stride",
    text: "MNE-Python: 0.5–50 Hz band-pass, notch, resample to 128 Hz, 2 s windows every 1 s, robust z-score per channel and recording, log band power in five bands." },
  { code: "G", title: "Directed graph per window", ann: "A[src → dst] ∈ [0, 1]",
    text: "An edge network scores every ordered pair from node embeddings, lagged cross-correlations and a per-window ridge VAR (conditional Granger evidence). It is trained by forecasting each channel's next sample from the gated past of the others." },
  { code: "S", title: "Spatial GNN", ann: "mix · prior + (1 − mix) · A",
    text: "Graph attention (PyTorch Geometric GATConv) over a mix of the learned graph and a distance prior from 10-20 electrode positions; directed diffusion convolution is the swappable alternative." },
  { code: "T", title: "Temporal Transformer", ann: "8 windows → the next",
    text: "A causal Transformer reads eight consecutive graph embeddings and predicts the next window's band power. It is trained only on normal data, so surprise means unusual." },
  { code: "A", title: "Anomaly score", ann: "z(error) + z(ΔA) + ½ z(fit)",
    text: "Robust z of the prediction error, plus how far the graph has moved from the preceding minute, plus the within-window forecast error; smoothed causally and thresholded on held-out normal data." },
  { code: "X", title: "Explanation", ann: "why this flag",
    text: "Propagation path (start at the channel whose outgoing dependencies grew most, then a walk over the edges that grew; edges between derivations sharing an electrode are excluded), edge masking, Integrated Gradients over channel × window × band, Transformer attention." },
];

export function Method() {
  return (
    <Page id="method" code="MTH" time="p. 2">
      <h2>Model the relationships, not only the signals.</h2>
      <p className="lede">
        Most EEG anomaly detectors score each channel's signal on its own. NEUROTRACE asks a different question: for every
        second, which channels' past helps predict which others, and does that pattern suddenly change?
      </p>
      <ol className="pipeline">
        {PIPELINE.map((p) => (
          <li key={p.code}>
            <span className="code">{p.code}</span>
            <div>
              <h4>{p.title}</h4>
              <p>{p.text}</p>
            </div>
            <span className="ann">{p.ann}</span>
          </li>
        ))}
      </ol>
      <h3>Checked against a ground truth first</h3>
      <p>
        A synthetic generator builds 19 damped oscillators on the 10-20 layout with a sparse network of lagged directed
        couplings, adds volume conduction, and injects events: a tree of new couplings opens from a seed channel, and in a
        quarter of events the seed also switches to a slow rhythm. Because every coupling is known, edge recovery and
        propagation order can be scored before the model touches real EEG. All splits are by subject, never by window.
      </p>
    </Page>
  );
}

function Bars({ rows, metric, title, note }: { rows: SummaryRow[]; metric: string; title: string; note: string }) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const data = rows.filter((r) => num(r, metric) != null).sort((a, b) => ORDER.indexOf(String(a.variant)) - ORDER.indexOf(String(b.variant)));
  if (!data.length) return null;
  const W = 560;
  const rowH = 26;
  const x0 = 250;
  const sx = (v: number) => x0 + (W - x0 - 46) * Math.max(0, Math.min(1, v));
  return (
    <figure style={{ margin: "18px 0 26px" }} onMouseLeave={() => setTip(null)}>
      <figcaption className="label" style={{ fontSize: 19, fontWeight: 600, marginBottom: 6 }}>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${data.length * rowH + 24}`} style={{ width: "100%", maxWidth: 720 }} role="img" aria-label={`${title}, one bar per model with 95% confidence intervals`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={sx(v)} x2={sx(v)} y1={0} y2={data.length * rowH} stroke={v === 0.5 ? "var(--ink-3)" : "var(--grid-major)"} strokeDasharray={v === 0.5 ? "3 3" : undefined} />
            <text x={sx(v)} y={data.length * rowH + 16} textAnchor="middle" fontFamily="var(--f-label)" fontSize={13} fill="var(--ink-3)">{v === 0.5 ? "0.5 chance" : v}</text>
          </g>
        ))}
        {data.map((r, i) => {
          const m = num(r, metric)!;
          const c = ci(r, metric) ?? 0;
          const n = nOf(r, metric) ?? Number(r.n_subjects);
          const v = String(r.variant);
          const isFull = v === "full";
          const isBase = v.startsWith("baseline_") || v === "channel_cnn_lstm";
          return (
            <g key={v} transform={`translate(0, ${i * rowH})`} onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text: `${LABELS[v] ?? v}: ${m.toFixed(3)} (95% CI ± ${c.toFixed(3)}, n = ${n} subjects)` })}>
              <rect x={0} y={0} width={W} height={rowH} fill="transparent" />
              <text x={x0 - 10} y={17} textAnchor="end" fontFamily="var(--f-label)" fontSize={14.5} fontWeight={isFull ? 700 : 500} fill={isBase ? "var(--ink-3)" : "var(--ink)"}>
                {isAblation(v) ? "– " : ""}{(LABELS[v] ?? v).replace("NEUROTRACE (learned graph + GAT + Transformer)", "NEUROTRACE (full)")}
              </text>
              <rect x={sx(0)} y={7} width={Math.max(0, sx(m) - sx(0))} height={12} rx={4} fill={isFull ? "var(--ink)" : isBase ? "var(--grid-major)" : "var(--blue)"} />
              <line x1={sx(m - c)} x2={sx(m + c)} y1={13} y2={13} stroke="var(--ink)" strokeWidth={1} />
              <line x1={sx(m - c)} x2={sx(m - c)} y1={9} y2={17} stroke="var(--ink)" strokeWidth={1} />
              <line x1={sx(m + c)} x2={sx(m + c)} y1={9} y2={17} stroke="var(--ink)" strokeWidth={1} />
              <text x={sx(m + c) + 6} y={17} fontFamily="var(--f-label)" fontSize={13.5} fill="var(--ink-2)" className="tnum">{m.toFixed(2)}</text>
            </g>
          );
        })}
      </svg>
      <p className="small">{note}</p>
      {tip && <div role="tooltip" style={{ position: "fixed", left: tip.x + 12, top: tip.y + 12, background: "var(--ink)", color: "var(--paper)", padding: "5px 8px", borderRadius: 3, font: "500 14px var(--f-label)", pointerEvents: "none", zIndex: 30 }}>{tip.text}</div>}
    </figure>
  );
}

function Table({ rows, synthetic }: { rows: SummaryRow[]; synthetic: boolean }) {
  const data = [...rows].sort((a, b) => ORDER.indexOf(String(a.variant)) - ORDER.indexOf(String(b.variant)));
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="table" style={{ minWidth: synthetic ? 1000 : 780 }}>
        <caption className="small" style={{ captionSide: "bottom", textAlign: "left", paddingTop: 8 }}>
          Mean ± half-width of the 95 % t-interval over test subjects; n = number of test subjects. Sensitivity and false
          alarms are pooled within each subject before averaging.
        </caption>
        <thead>
          <tr>
            <th>Model</th>
            <th className="num">n</th>
            <th className="num">AUROC</th>
            <th className="num">AUPRC</th>
            <th className="num">Sensitivity</th>
            <th className="num">False alarms / h</th>
            <th className="num">Latency (s)</th>
            {synthetic && <th className="num">Edge AUROC</th>}
            {synthetic && <th className="num">Path seed hit</th>}
          </tr>
        </thead>
        <tbody>
          {data.map((r) => {
            const v = String(r.variant);
            return (
              <tr key={v} className={v === "full" ? "hl" : ""}>
                <td style={{ paddingLeft: isAblation(v) ? 28 : 8 }}>{LABELS[v] ?? v}</td>
                <td className="num">{String(r.n_subjects ?? "—")}</td>
                <td className="num">{fci(r, "auroc")}</td>
                <td className="num">{fci(r, "auprc")}</td>
                <td className="num">{fci(r, "sensitivity")}</td>
                <td className="num">{fci(r, "fa_per_hour", 1)}</td>
                <td className="num">{fci(r, "median_latency_s", 1)}</td>
                {synthetic && <td className="num">{fci(r, "edge_auroc")}</td>}
                {synthetic && <td className="num">{fci(r, "path_seed_hit")}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Results({ index }: { index: Index }) {
  const syn = index.summary.synthetic ?? [];
  const chb = index.summary.chbmit ?? [];
  const row = (rows: SummaryRow[], v: string) => rows.find((x) => x.variant === v);
  const val = (rows: SummaryRow[], v: string, m: string) => {
    const r = row(rows, v);
    return r ? fci(r, m) : "—";
  };
  const bestBase = (rows: SummaryRow[], m: string) =>
    rows.filter((r) => String(r.variant).startsWith("baseline_") || r.variant === "channel_cnn_lstm").reduce<SummaryRow | null>((b, r) => (num(r, m) != null && (b == null || num(r, m)! > num(b, m)!) ? r : b), null);
  const synBest = bestBase(syn, "auroc");
  const chbBest = bestBase(chb, "auroc");
  const synN = row(syn, "full")?.n_subjects;
  const chbN = row(chb, "full")?.n_subjects;
  return (
    <Page id="results" code="RES" time="p. 3">
      <h2>What held up, and what did not.</h2>
      <p className="lede">
        Every number is a mean over held-out test subjects with its 95 % confidence interval, written by{" "}
        <code>scripts/evaluate.py</code> to <code>results/</code>. Baselines are scored with the same thresholds and event
        matching as the model, and are shown even where they win.
      </p>

      <h3>Synthetic data: is the learned graph right?</h3>
      {syn.length ? (
        <>
          <p>
            Over {String(synN ?? "—")} held-out synthetic subjects, the learned edges recover the injected directed couplings
            with an edge AUROC of <strong>{val(syn, "full", "edge_auroc")}</strong>; per-window conditional Granger causality, the
            natural estimator for data generated by a linear autoregressive model, reaches {val(syn, "baseline_granger", "edge_auroc")}.
            Without the VAR evidence the edge module reaches {val(syn, "no_var_evidence", "edge_auroc")}. For event detection the full
            model reaches {val(syn, "full", "auroc")} window AUROC; the best baseline ({LABELS[String(synBest?.variant)] ?? "—"}) reaches{" "}
            {synBest ? fci(synBest, "auroc") : "—"}.
          </p>
          <Bars rows={syn} metric="edge_auroc" title="Edge recovery, synthetic (AUROC over all directed pairs)" note="Per-window learned adjacency vs the couplings active in that window; Granger, PDC and DTF on the same windows. Whiskers: 95 % CI over test subjects." />
          <Bars rows={syn} metric="auroc" title="Event detection, synthetic (window AUROC)" note="Self-supervised variants never see labels; the supervised head does, and is an upper reference rather than a fair competitor." />
          <Table rows={syn} synthetic />
          <h3>Reading the event columns and the paths</h3>
          <p>
            The event metrics are not one operating point. Each detector's threshold is the 99th percentile of its own scores
            on a held-out subject's normal windows: the self-supervised models end up quiet ({val(syn, "full", "fa_per_hour")} false
            alarms per hour) and catch few events ({val(syn, "full", "sensitivity")}), while the connectivity baselines catch more at
            several false alarms per hour. Compare AUROC and AUPRC across rows; compare sensitivity only together with its false-alarm rate.
          </p>
          <p>
            Propagation paths start from the channel whose outgoing dependencies grew most. From the learned graph that channel is
            the true seed in {val(syn, "full", "path_seed_hit")} of events; from Granger in {val(syn, "baseline_granger", "path_seed_hit")}.
            The learned edges are bounded gates, so a strong new coupling and a weak one look alike, and the start rule loses the
            seed. A rule that also weighed band-power rise looked better on the first, easier benchmark and did not hold up here
            ({val(syn, "full", "path_seed_hit_hybrid")}); it is reported in the CSV and not used.
          </p>
        </>
      ) : (
        <p className="empty">The synthetic study is being re-run after an evaluation audit; this section fills in from results/synthetic/summary.csv.</p>
      )}

      <h3>CHB-MIT: seizures in held-out patients</h3>
      {chb.length ? (
        <>
          <p>
            Leave-one-subject-out over {String(chbN ?? "—")} patients. The self-supervised model reaches a window AUROC of{" "}
            <strong>{val(chb, "full", "auroc")}</strong>; the best baseline ({LABELS[String(chbBest?.variant)] ?? "—"}) reaches{" "}
            {chbBest ? fci(chbBest, "auroc") : "—"}. Seizures are rare, which is why AUPRC and false alarms per hour matter more
            than AUROC.
          </p>
          <p>
            The simplest detector wins. Band-power deviation reaches the best window AUROC and AUPRC of the label-free
            detectors and catches the most seizures, at about four false alarms per hour. The full model is the quietest
            ({val(chb, "full", "fa_per_hour")} false alarms per hour) and misses most seizures at its threshold
            ({val(chb, "full", "sensitivity")}). The learned graph does carry signal: the same model with no graph or with the
            distance graph alone drops to {val(chb, "no_graph", "auroc")} and {val(chb, "static_graph", "auroc")}, mostly
            because of two patients. The other ablations (diffusion instead of GAT, no Transformer, no VAR evidence, masked
            objective) land between {val(chb, "no_transformer", "auroc")} and {val(chb, "masked_objective", "auroc")}: on real data
            none of those choices is measurably better. The connectivity baselines reach 0.79–0.85, Granger highest. The supervised head ({val(chb, "supervised_head", "auroc")}) is trained on the other patients'
            seizure labels and is an upper reference, not a fair competitor. The recordings in the explorer belong to the
            first held-out patient (chb05), scored by the model that never saw that patient.
          </p>
          <Bars rows={chb} metric="auroc" title="Seizure detection, CHB-MIT (window AUROC)" note="Each patient held out once. Whiskers are 95 % confidence intervals over patients; with this few patients they are wide." />
          <Table rows={chb} synthetic={false} />
        </>
      ) : (
        <p className="empty">No CHB-MIT results are reported yet: the leave-one-subject-out run is in progress. This section fills in from results/chbmit/summary.csv.</p>
      )}
    </Page>
  );
}

export function Limitations() {
  return (
    <Page id="limits" code="LIM" time="p. 4">
      <h2>Read this before reading the map.</h2>
      <ul className="caveats">
        <li><strong>Not a medical device.</strong>This is a research and education prototype. It has not been validated clinically and must not be used to diagnose, monitor or treat anyone.</li>
        <li><strong>Arrows are dependencies, not causes.</strong>An edge means one channel's past helps predict another's. Shared inputs, filtering, reference choice and volume conduction all create such edges without any causal pathway.</li>
        <li><strong>Electrodes are not brain regions.</strong>Each scalp electrode mixes many cortical sources; CHB-MIT channels are bipolar derivations drawn at the midpoint of their two electrodes. Source reconstruction is an optional extension, with its own errors.</li>
        <li><strong>Timing is at window resolution.</strong>Paths are ordered at the 1 s stride. Real and synthetic propagation delays are tens of milliseconds, so a path is a walk order, not measured arrival times, and on real EEG it cannot be checked at all.</li>
        <li><strong>Small real-data cohort.</strong>Six CHB-MIT patients (paediatric, one centre, seizure files plus three seizure-free hours each). Confidence intervals are wide; no claim is made about other populations or the TUH corpus.</li>
        <li><strong>The sparsity setting matters.</strong>Edge recovery depends on the L1 weight on the learned graph: 0.02–0.03 works, 0.05 collapses every gate. It was tuned on synthetic data and transferred unchanged.</li>
        <li><strong>The synthetic world is linear.</strong>A VAR-style generator favours Granger-type estimators, so synthetic edge recovery is a sanity check for the model, not a benchmark it should be expected to win.</li>
      </ul>
      <h3>Run it yourself</h3>
      <pre className="cmd">{`git clone https://github.com/Lawl1et-1/neurotrace && cd neurotrace
pip install -e ".[app,dev]"
make test                                      # unit tests, CPU
python scripts/evaluate.py experiment=full     # synthetic ground-truth study
make data && make eval-chbmit                  # CHB-MIT, leave-one-subject-out
make app                                       # this visualizer on localhost`}</pre>
    </Page>
  );
}
