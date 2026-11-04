import { useState } from "react";
import { fmtTime, type Index, type Segment } from "../data";

interface Props {
  seg: Segment;
  index: Index;
  selected: number | null;
  onSelect: (i: number) => void;
  reveal: number;
  setHot: (c: number | null) => void;
}

/** Sequential single-hue ramp (paper -> ink blue) for magnitudes. */
const ramp = (v: number) => {
  const a = [244, 247, 245];
  const b = [38, 93, 150];
  const k = Math.max(0, Math.min(1, v));
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * k)).join(",")})`;
};

export default function EventReport({ seg, index, selected, onSelect, reveal, setHot }: Props) {
  const [pathKind, setPathKind] = useState<"graph" | "signal">("graph");
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const names = seg.ch_names;

  if (!seg.events.length)
    return (
      <>
        <h2>No event crossed the threshold in this record.</h2>
        <p className="lede">
          The alarm threshold is fixed on a different subject's normal data before the test subject is seen, so a quiet
          record stays quiet. {seg.labels.length ? "This one has a labelled event the model missed: a false negative, counted as such in the results." : ""}
        </p>
      </>
    );

  if (selected == null)
    return (
      <>
        <h2>Pick an event on the timeline to read its report.</h2>
        <div className="actions" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {seg.events.map((e, i) => (
            <button key={i} className="btn ghost" onClick={() => onSelect(i)}>
              Event {i + 1} · {fmtTime(e.onset_s)}
            </button>
          ))}
        </div>
      </>
    );

  const ev = seg.events[selected];
  const path = pathKind === "graph" ? ev.graph_path : ev.signal_path;
  const truth = seg.truth?.events.find((te) => Math.abs(te.onset_s - ev.onset_s) < 15) ?? null;
  const matched = truth ? path.filter((p) => truth.order.includes(p.ch)).length : 0;
  const igMax = Math.max(1e-9, ...ev.ig_bands.flat().map(Math.abs));
  const igRows = ev.ig_bands.map((row, c) => ({ c, row, tot: row.reduce((a, b) => a + Math.abs(b), 0) })).sort((a, b) => b.tot - a.tot).slice(0, 8);
  const mask = new Map(ev.edge_masking.map((m) => [`${m.src}-${m.dst}`, m.effect ?? 0]));
  const attMax = Math.max(1e-9, ...(ev.attention ?? [0]));
  const stride = seg.stride_s;

  return (
    <div onMouseLeave={() => setTip(null)}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 18 }} role="tablist" aria-label="Detected events">
        {seg.events.map((e, i) => (
          <button key={i} role="tab" aria-selected={i === selected} className={`btn ${i === selected ? "" : "ghost"}`} onClick={() => onSelect(i)}>
            Event {i + 1} · {fmtTime(e.onset_s)}
          </button>
        ))}
      </div>
      <h2>
        Event {selected + 1}: flagged at {fmtTime(ev.onset_s)}, starting from {names[ev.graph_path[0]?.ch ?? 0]}.
      </h2>
      <p className="lede">
        The anomaly score peaked at {ev.peak_score.toFixed(1)} against a threshold of {seg.threshold.toFixed(1)}.
        {truth
          ? ` The injected event really began at ${names[truth.seed]}, at ${fmtTime(truth.onset_s)}${ev.graph_path[0]?.ch === truth.seed ? ", so the model's starting point is right." : ", so the model's starting point is wrong here."}`
          : seg.dataset === "chbmit"
            ? " On real EEG there is no ground truth for the path: read it as where the model saw the dependency structure change first."
            : ""}
      </p>
      {ev.hub_warning && ev.hub != null && (
        <p className="warn" role="note">
          <strong>Hub warning.</strong> {names[ev.hub]} takes part in {Math.round((ev.hub_share ?? 0) * 100)} % of this event's graph
          increase (an even spread would give each channel about {Math.round(100 / names.length)} %). One channel dominating
          more often means an electrode problem, an artefact on one derivation or a reference effect than a spread across the
          brain.
        </p>
      )}
      {(ev.shared_share ?? 0) > 0 && (
        <p className="small" style={{ marginTop: -6, marginBottom: 14 }}>
          {Math.round((ev.shared_share ?? 0) * 100)} % of the increase sat on pairs of derivations that share an electrode;
          those edges are left out of the path and the edge table.
        </p>
      )}

      <div className="report">
        <div className="col-path">
          <h3>
            Propagation path <span className="note">walk order, not arrival times</span>
          </h3>
          <div className="toggle" role="group" aria-label="Path estimate" style={{ marginBottom: 10 }}>
            <button aria-pressed={pathKind === "graph"} onClick={() => setPathKind("graph")}>From the learned graph</button>
            <button aria-pressed={pathKind === "signal"} onClick={() => setPathKind("signal")}>From band power</button>
          </div>
          <ol className="path-list">
            {path.slice(0, 10).map((p, i) => (
              <li key={`${p.ch}-${i}`} className={pathKind === "signal" || i < reveal ? "lit" : ""} onMouseEnter={() => setHot(p.ch)} onMouseLeave={() => setHot(null)}>
                <span className="n">{i + 1}</span>
                <span className="ch">{names[p.ch]}</span>
                <span className="via">{p.via != null ? `via ${names[p.via]}` : i === 0 ? "start" : ""}{truth && (truth.order.includes(p.ch) ? " · in true path" : " · not in true path")}</span>
                <span className="t tnum">{i === 0 ? fmtTime(p.t) : ""}</span>
              </li>
            ))}
          </ol>
          {truth && (
            <p className="small" style={{ marginTop: 10 }}>
              True order: {truth.order.map((c) => names[c]).join(" → ")}. {matched} of {truth.order.length} channels recovered. The walk starts at the channel whose
              outgoing dependencies grew most, then follows the edges that grew most. The injected
              delays are 15–40 ms, far below the {stride} s window step, so only the order is testable.
            </p>
          )}
          {!truth && <p className="small" style={{ marginTop: 10 }}>The walk is ordered by dependency strength at {stride} s resolution: a chain of statistical dependencies, not a measured conduction route or arrival sequence.</p>}
        </div>

        <div className="col-edges">
          <h3>
            Edges that changed most <span className="note">vs the minute before</span>
          </h3>
          <table className="table">
            <thead>
              <tr>
                <th>Dependency</th>
                <th className="num">Change</th>
                <th className="num" title="How much the prediction error drops when this edge is removed from the graph">Masking effect</th>
              </tr>
            </thead>
            <tbody>
              {ev.top_edges.slice(0, 8).map((e) => {
                const inTruth = truth?.edges.some(([s, d]) => s === e.src && d === e.dst);
                return (
                  <tr key={`${e.src}-${e.dst}`} className={inTruth ? "hl" : ""} onMouseEnter={() => setHot(e.src)} onMouseLeave={() => setHot(null)}>
                    <td>
                      <span className="label" style={{ fontSize: 17, fontWeight: 600 }}>{names[e.src]} → {names[e.dst]}</span>
                      {truth && <span className="small">{inTruth ? "  injected" : ""}</span>}
                    </td>
                    <td className="num">{(e.delta ?? 0) >= 0 ? "+" : ""}{(e.delta ?? 0).toFixed(3)}</td>
                    <td className="num">{mask.has(`${e.src}-${e.dst}`) ? (mask.get(`${e.src}-${e.dst}`)! * 1000).toFixed(2) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="small" style={{ marginTop: 8 }}>
            Masking effect ×10⁻³: drop in the prediction error when the edge is removed from the whole input sequence. Small values mean the flag does not rest on that single edge.{" "}
            {truth && "Shaded rows are couplings that were really injected."}
          </p>
        </div>

        <div className="col-ig">
          <h3>
            Where the flag comes from <span className="note">Integrated Gradients</span>
          </h3>
          <svg viewBox={`0 0 ${110 + index.bands.length * 70} ${igRows.length * 30 + 26}`} style={{ width: "100%", maxWidth: 520 }} role="img" aria-label="Attribution of the prediction error by channel and frequency band">
            {index.bands.map((b, j) => (
              <text key={b} x={110 + j * 70 + 33} y={14} textAnchor="middle" fontFamily="var(--f-label)" fontSize={14} fill="var(--ink-3)">{b}</text>
            ))}
            {igRows.map(({ c, row }, r) => (
              <g key={c} transform={`translate(0, ${22 + r * 30})`}>
                <text x={100} y={19} textAnchor="end" fontFamily="var(--f-label)" fontWeight={600} fontSize={16} fill="var(--ink)">{names[c]}</text>
                {row.map((v, j) => (
                  <rect
                    key={j} x={110 + j * 70} y={2} width={66} height={26} rx={2}
                    fill={ramp(Math.abs(v) / igMax)}
                    stroke="var(--paper)" strokeWidth={2}
                    onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text: `${names[c]} · ${index.bands[j]}: ${v >= 0 ? "+" : ""}${v.toExponential(2)}` })}
                  />
                ))}
              </g>
            ))}
          </svg>
          <p className="small">
            Attribution magnitude of the prediction error to each channel's band power, summed over the {ev.ig_channel_time.length} input windows; the eight largest channels are shown.
            Convergence error {ev.ig_delta.toExponential(1)}
            {ev.ig_steps != null && ` after ${ev.ig_steps} steps`}
            {ev.ig_converged === false && " (did not reach the 1e-2 target: read with care)"}. The GNN's edge choice is held at
            this window's along the integration path. It explains the model, not the brain.
          </p>
        </div>

        <div className="col-att">
          <h3>
            What the Transformer looked at <span className="note">attention</span>
          </h3>
          {ev.attention ? (
            <>
              <svg viewBox={`0 0 ${ev.attention.length * 44} 70`} style={{ width: "100%", maxWidth: 420 }} role="img" aria-label="Attention of the last position over the preceding windows">
                {ev.attention.map((a, i) => (
                  <g key={i} transform={`translate(${i * 44}, 0)`}>
                    <rect x={2} y={4} width={40} height={36} rx={2} fill={ramp(a / attMax)} onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text: `window t${i - ev.attention!.length + 1}: ${(a * 100).toFixed(1)}%` })} />
                    <text x={22} y={58} textAnchor="middle" fontFamily="var(--f-label)" fontSize={13} fill="var(--ink-3)">{i === ev.attention!.length - 1 ? "t" : `t${i - ev.attention!.length + 1}`}</text>
                  </g>
                ))}
              </svg>
              <p className="small">Last layer, mean over heads: how much the prediction for window t weighs each earlier window (1 window = {stride} s).</p>
            </>
          ) : (
            <p className="small">This model variant has no temporal attention.</p>
          )}
        </div>
      </div>
      {tip && (
        <div role="tooltip" style={{ position: "fixed", left: tip.x + 12, top: tip.y + 12, background: "var(--ink)", color: "var(--paper)", padding: "5px 8px", borderRadius: 3, font: "500 14px var(--f-label)", pointerEvents: "none", zIndex: 30 }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}
