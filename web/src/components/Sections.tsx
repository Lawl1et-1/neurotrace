import { useState } from "react";
import type { Index, SummaryRow } from "../data";

const LABELS: Record<string, string> = {
  full: "NEUROTRACE (learned graph + GAT + Transformer)",
  diffusion_gnn: "· diffusion GNN instead of GAT",
  static_graph: "· distance-prior graph only",
  no_graph: "· no graph (channels mixed only at readout)",
  no_transformer: "· no Transformer",
  no_var_evidence: "· edge module without VAR evidence",
  masked_objective: "· masked-window objective",
  supervised_head: "· + supervised head (uses labels)",
  channel_cnn_lstm: "Per-channel CNN-LSTM forecaster",
  baseline_bandpower_dev: "Band-power deviation",
  baseline_pearson: "Pearson-change",
  baseline_coherence: "Coherence-change",
  baseline_granger: "Granger (per window)",
  baseline_pdc: "PDC (per window)",
  baseline_dtf: "DTF (per window)",
};
const ORDER = Object.keys(LABELS);

const num = (r: SummaryRow, k: string) => (typeof r[`${k}_mean`] === "number" ? (r[`${k}_mean`] as number) : null);
const f = (v: number | null, d = 2) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));

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

export function Method() {
  return (
    <Page id="method" code="MTH" time="p. 2">
      <h2>Model the relationships, not only the signals.</h2>
      <p className="lede">
        Most EEG anomaly detectors score each channel's signal on its own. NEUROTRACE asks a different question: for every
        second, which channels' past helps predict which others, and does that pattern suddenly change?
      </p>
      <div className="steps">
        <div>
          <h4>1 · Windows</h4>
          <p>MNE-Python: 0.5–50 Hz band-pass, notch, resample to 128 Hz, 2 s windows every 1 s, robust z-score per channel and recording. Log band power in five bands.</p>
        </div>
        <div>
          <h4>2 · Directed graph per window</h4>
          <p>An edge network scores every ordered pair from node embeddings, lagged cross-correlations and a per-window ridge VAR (conditional Granger evidence). It is trained by forecasting each channel's next sample from the gated past of the others.</p>
          <span className="sig">A[src → dst] ∈ [0, 1]</span>
        </div>
        <div>
          <h4>3 · Spatial GNN</h4>
          <p>Graph attention (PyTorch Geometric GATConv) over a mix of the learned graph and a distance prior built from 10-20 electrode positions. Diffusion convolution is the swappable alternative.</p>
        </div>
        <div>
          <h4>4 · Temporal Transformer</h4>
          <p>A causal Transformer reads eight consecutive graph embeddings and predicts the next window's band power. It is trained only on normal data, so surprise means unusual.</p>
        </div>
        <div>
          <h4>5 · Anomaly score</h4>
          <p>Robust z of prediction error + graph change |A_t − A_t−1| + within-window forecast error, smoothed causally. The threshold is set on held-out normal data.</p>
        </div>
        <div>
          <h4>6 · Explanation</h4>
          <p>Propagation path (greedy walk over the edges that grew), edge masking, Integrated Gradients over channel × window × band (Captum), Transformer attention.</p>
        </div>
      </div>
      <h3>Checked against a ground truth first</h3>
      <p>
        A synthetic generator builds 19 damped oscillators on the 10-20 layout with a sparse network of lagged directed
        couplings, adds mild volume conduction, and injects events: a seed channel switches to a slow rhythm and a tree of
        new couplings carries it outward. Because every coupling is known, edge recovery and propagation order can be scored
        before the model touches real EEG. All splits are by subject, never by window.
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
  const sx = (v: number) => x0 + (W - x0 - 40) * v;
  return (
    <figure style={{ margin: "18px 0 26px" }} onMouseLeave={() => setTip(null)}>
      <figcaption className="label" style={{ fontSize: 19, fontWeight: 600, marginBottom: 6 }}>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${data.length * rowH + 24}`} style={{ width: "100%", maxWidth: 720 }} role="img" aria-label={`${title}, one bar per model`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={sx(v)} x2={sx(v)} y1={0} y2={data.length * rowH} stroke={v === 0.5 ? "var(--ink-3)" : "var(--grid-major)"} strokeDasharray={v === 0.5 ? "3 3" : undefined} />
            <text x={sx(v)} y={data.length * rowH + 16} textAnchor="middle" fontFamily="var(--f-label)" fontSize={13} fill="var(--ink-3)">{v === 0.5 ? "0.5 chance" : v}</text>
          </g>
        ))}
        {data.map((r, i) => {
          const m = num(r, metric)!;
          const sd = typeof r[`${metric}_std`] === "number" ? (r[`${metric}_std`] as number) : 0;
          const v = String(r.variant);
          const isFull = v === "full";
          const isBase = v.startsWith("baseline_") || v === "channel_cnn_lstm";
          return (
            <g key={v} transform={`translate(0, ${i * rowH})`} onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text: `${LABELS[v] ?? v}: ${m.toFixed(3)} ± ${sd.toFixed(3)} (n=${r.n_subjects})` })}>
              <rect x={0} y={0} width={W} height={rowH} fill="transparent" />
              <text x={x0 - 10} y={17} textAnchor="end" fontFamily="var(--f-label)" fontSize={14.5} fontWeight={isFull ? 700 : 500} fill={isBase ? "var(--ink-3)" : "var(--ink)"}>
                {(LABELS[v] ?? v).replace("NEUROTRACE (learned graph + GAT + Transformer)", "NEUROTRACE (full)")}
              </text>
              <rect x={sx(0)} y={7} width={Math.max(0, sx(Math.max(0, m)) - sx(0))} height={12} rx={4} fill={isFull ? "var(--ink)" : isBase ? "var(--grid-major)" : "var(--blue)"} />
              <line x1={sx(Math.max(0, m - sd))} x2={sx(Math.min(1, m + sd))} y1={13} y2={13} stroke="var(--ink)" strokeWidth={1} />
              <text x={sx(Math.min(1, Math.max(0, m) + sd)) + 6} y={17} fontFamily="var(--f-label)" fontSize={13.5} fill="var(--ink-2)" className="tnum">{m.toFixed(2)}</text>
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
      <table className="table" style={{ minWidth: 720 }}>
        <thead>
          <tr>
            <th>Model</th>
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
          {data.map((r) => (
            <tr key={String(r.variant)} className={r.variant === "full" ? "hl" : ""}>
              <td>{LABELS[String(r.variant)] ?? r.variant}</td>
              <td className="num">{f(num(r, "auroc"))}</td>
              <td className="num">{f(num(r, "auprc"))}</td>
              <td className="num">{f(num(r, "sensitivity"))}</td>
              <td className="num">{f(num(r, "fa_per_hour"), 1)}</td>
              <td className="num">{f(num(r, "median_latency_s"), 0)}</td>
              {synthetic && <td className="num">{f(num(r, "edge_auroc"))}</td>}
              {synthetic && <td className="num">{f(num(r, "path_seed_hit"))}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Results({ index }: { index: Index }) {
  const syn = index.summary.synthetic ?? [];
  const chb = index.summary.chbmit ?? [];
  const get = (rows: SummaryRow[], v: string, m: string) => {
    const r = rows.find((x) => x.variant === v);
    return r ? num(r, m) : null;
  };
  const bestBase = (rows: SummaryRow[], m: string) =>
    rows.filter((r) => String(r.variant).startsWith("baseline_") || r.variant === "channel_cnn_lstm").reduce<SummaryRow | null>((b, r) => (num(r, m) != null && (b == null || num(r, m)! > num(b, m)!) ? r : b), null);
  const synBest = bestBase(syn, "auroc");
  const chbBest = bestBase(chb, "auroc");
  return (
    <Page id="results" code="RES" time="p. 3">
      <h2>What held up, and what did not.</h2>
      <p className="lede">
        Every number below is a mean over held-out subjects, written by <code>scripts/evaluate.py</code> to <code>results/</code>.
        Baselines are reported next to the model, including where they win.
      </p>

      <h3>Synthetic data: is the learned graph right?</h3>
      {syn.length ? (
        <>
          <p>
            The learned edges recover the injected directed couplings with an edge AUROC of <strong>{f(get(syn, "full", "edge_auroc"))}</strong>,
            close to per-window conditional Granger causality ({f(get(syn, "baseline_granger", "edge_auroc"))}), which is the right
            estimator for data generated by a linear autoregressive model. Without the VAR evidence the edge module drops to{" "}
            {f(get(syn, "no_var_evidence", "edge_auroc"))}: correlations alone cannot separate direct couplings from shared inputs.
            For anomaly detection the graph adds little here: the best simple baseline ({LABELS[String(synBest?.variant)] ?? "—"}) reaches
            {" "}{f(synBest ? num(synBest, "auroc") : null)} window AUROC against {f(get(syn, "full", "auroc"))} for the full model.
          </p>
          <Bars rows={syn} metric="edge_auroc" title="Edge recovery, synthetic (AUROC over all directed pairs)" note="Per-window learned adjacency vs the couplings active in that window. Granger, PDC and DTF are computed on the same windows." />
          <Bars rows={syn} metric="auroc" title="Event detection, synthetic (window AUROC)" note="Self-supervised variants never see labels; the supervised head does, and is an upper reference rather than a fair competitor." />
          <Table rows={syn} synthetic />
        </>
      ) : (
        <p className="empty">Synthetic evaluation still running; this table fills in from results/synthetic/summary.csv.</p>
      )}

      <h3>CHB-MIT: seizures in held-out patients</h3>
      {chb.length ? (
        <>
          <p>
            Leave-one-subject-out over {String(chb[0]?.n_subjects ?? "")} patients. The self-supervised model reaches a window AUROC of{" "}
            <strong>{f(get(chb, "full", "auroc"))}</strong> (best baseline: {LABELS[String(chbBest?.variant)] ?? "—"},{" "}
            {f(chbBest ? num(chbBest, "auroc") : null)}). Seizures are rare (low prevalence), which is why AUPRC and false alarms
            per hour matter more than AUROC.
          </p>
          <Bars rows={chb} metric="auroc" title="Seizure detection, CHB-MIT (window AUROC)" note="Six subjects, each held out once. Small n: differences of a few hundredths are within subject-to-subject spread (error bars = 1 sd)." />
          <Table rows={chb} synthetic={false} />
        </>
      ) : (
        <p className="empty">CHB-MIT cross-validation still running; this table fills in from results/chbmit/summary.csv.</p>
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
        <li><strong>Timing is at window resolution.</strong>Paths are ordered at the 1 s stride. Real and synthetic propagation delays are tens of milliseconds, so only the order of a path can be checked, and on real EEG not even that.</li>
        <li><strong>Small real-data cohort.</strong>Six CHB-MIT patients (paediatric, one centre, mostly seizure files plus three seizure-free hours each). Results will move with more subjects; no claim is made about other populations or the TUH corpus.</li>
        <li><strong>The sparsity setting matters.</strong>Edge recovery depends on the L1 weight on the learned graph: 0.02–0.03 works, 0.05 collapses every gate. It was tuned on synthetic data and transferred unchanged.</li>
        <li><strong>The synthetic world is linear.</strong>A VAR-style generator favours Granger-type estimators, so synthetic edge recovery is an easy test for them and a sanity check for the model, not a benchmark.</li>
      </ul>
      <h3>Run it yourself</h3>
      <pre className="cmd">{`git clone https://github.com/Lawl1et-1/neurotrace && cd neurotrace
pip install -e ".[app,dev]"
make test                          # unit tests, CPU
python scripts/evaluate.py experiment=full     # synthetic ground-truth study
make data && python scripts/evaluate.py experiment=chbmit
make app                           # this visualizer on localhost`}</pre>
    </Page>
  );
}
