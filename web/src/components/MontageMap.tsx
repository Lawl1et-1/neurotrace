import { useMemo } from "react";
import type { Edge, Segment, TruthEvent } from "../data";

interface Props {
  seg: Segment;
  edges: Edge[];
  litPath: { ch: number; via?: number | null }[]; // propagation steps revealed so far
  hot: number | null;
  setHot: (ch: number | null) => void;
  truth: TruthEvent | null; // ground-truth event active at this time (synthetic only)
  showTruth: boolean;
  showChange: boolean;
}

const R = 0.085; // node radius in head units

/** Quadratic arc from a to b, bowed to the left of the direction so i->j and j->i separate. */
function arc(a: [number, number], b: [number, number], bow = 0.18) {
  const [x1, y1] = a;
  const [x2, y2] = b;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  // shorten both ends so the line starts and stops at the node rims
  const sx = x1 + ux * R * 1.15;
  const sy = y1 + uy * R * 1.15;
  const ex = x2 - ux * R * 1.6;
  const ey = y2 - uy * R * 1.6;
  const mx = (sx + ex) / 2 - uy * len * bow;
  const my = (sy + ey) / 2 + ux * len * bow;
  return `M${sx.toFixed(3)},${sy.toFixed(3)} Q${mx.toFixed(3)},${my.toFixed(3)} ${ex.toFixed(3)},${ey.toFixed(3)}`;
}

export default function MontageMap({ seg, edges, litPath, hot, setHot, truth, showTruth, showChange }: Props) {
  // SVG y grows downward; the topomap has the nose at +y.
  const pos = useMemo(() => seg.pos2d.map(([x, y]) => [x, -y] as [number, number]), [seg]);
  const bipolar = seg.ch_names[0].includes("-");
  const wMax = Math.max(1e-6, ...edges.map((e) => e.w));
  const chMax = Math.max(1e-6, ...edges.map((e) => Math.abs(e.change)));
  const litSet = new Set(litPath.map((p) => p.ch));
  const litEdges = litPath.filter((p) => p.via != null).map((p) => [p.via as number, p.ch] as [number, number]);

  const isChanged = (e: Edge) => showChange && e.change > 0.35 * chMax && e.change > 0.04;
  const touches = (e: { src: number; dst: number }) => hot == null || e.src === hot || e.dst === hot;

  return (
    <svg viewBox="-1.32 -1.28 2.64 2.52" role="img" aria-label={`Directed dependency map for ${seg.record}: ${edges.length} strongest learned edges`}>
      <defs>
        {(["ink", "blue", "red", "green"] as const).map((k) => (
          <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse">
            <path d="M0,1 L9,5 L0,9 z" fill={`var(--${k})`} />
          </marker>
        ))}
      </defs>

      {/* head outline, nose and ears in pen ink */}
      <g fill="none" stroke="var(--ink)" strokeWidth="0.012" opacity="0.8">
        <circle cx="0" cy="0" r="1.0" />
        <path d="M-0.1,-0.995 L0,-1.13 L0.1,-0.995" />
        <path d="M-1.0,-0.12 q-0.09,0.02 -0.09,0.12 q0,0.1 0.09,0.12" />
        <path d="M1.0,-0.12 q0.09,0.02 0.09,0.12 q0,0.1 -0.09,0.12" />
      </g>
      <g stroke="var(--grid-major)" strokeWidth="0.006" fill="none">
        <circle cx="0" cy="0" r="0.5" />
        <line x1="-1" y1="0" x2="1" y2="0" />
        <line x1="0" y1="-1" x2="0" y2="1" />
      </g>

      {/* learned edges: width and opacity = strength; red = grew vs the previous minute */}
      <g>
        {edges.map((e) => {
          const red = isChanged(e);
          const op = touches(e) ? 0.25 + 0.75 * (e.w / wMax) : 0.06;
          return (
            <path
              key={`${e.src}-${e.dst}`}
              d={arc(pos[e.src], pos[e.dst])}
              fill="none"
              stroke={red ? "var(--red)" : "var(--blue)"}
              strokeWidth={0.008 + 0.03 * (e.w / wMax)}
              strokeOpacity={op}
              markerEnd={`url(#arrow-${red ? "red" : "blue"})`}
            >
              <title>{`${seg.ch_names[e.src]} → ${seg.ch_names[e.dst]}: dependency ${e.w.toFixed(2)} (${e.change >= 0 ? "+" : ""}${e.change.toFixed(2)} vs previous minute)`}</title>
            </path>
          );
        })}
      </g>

      {/* ground truth of the injected event (synthetic data only), dashed green */}
      {showTruth && truth && (
        <g>
          {truth.edges.map(([s, d]) => (
            <path key={`t${s}-${d}`} d={arc(pos[s], pos[d], -0.22)} fill="none" stroke="var(--green)" strokeWidth="0.014" strokeDasharray="0.04 0.03" markerEnd="url(#arrow-green)" />
          ))}
        </g>
      )}

      {/* propagation path revealed step by step */}
      <g>
        {litEdges.map(([s, d], i) => (
          <path key={`p${i}`} d={arc(pos[s], pos[d])} fill="none" stroke="var(--red)" strokeWidth="0.03" markerEnd="url(#arrow-red)" className="path-draw" />
        ))}
      </g>

      {/* electrodes */}
      <g>
        {pos.map(([x, y], i) => {
          const lit = litSet.has(i);
          const isSeed = litPath[0]?.ch === i;
          const isTruthSeed = showTruth && truth?.seed === i;
          return (
            <g
              key={i}
              transform={`translate(${x},${y})`}
              onMouseEnter={() => setHot(i)}
              onMouseLeave={() => setHot(null)}
              onFocus={() => setHot(i)}
              onBlur={() => setHot(null)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setHot(i);
                }
                if (e.key === "Escape") setHot(null);
              }}
              className="node"
              tabIndex={0}
              role="button"
              aria-label={`${seg.ch_names[i]}${lit ? ", on the propagation path" : ""}`}
              style={{ cursor: "pointer" }}
            >
              {/* keyboard focus ring: drawn on the node itself, shown only on :focus-visible (see styles.css) */}
              <circle className="focus-ring" r={R * 1.4} fill="none" stroke="var(--red)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              {isTruthSeed && <circle r={R * 1.55} fill="none" stroke="var(--green)" strokeWidth="0.012" strokeDasharray="0.03 0.02" />}
              <circle r={R} fill={lit ? "var(--red)" : hot === i ? "var(--ink)" : "var(--paper)"} stroke={lit ? "var(--red)" : "var(--ink)"} strokeWidth={isSeed ? 0.024 : 0.012} />
              <text
                y={bipolar ? 0.017 : 0.026}
                textAnchor="middle"
                fontFamily="var(--f-label)"
                fontWeight={600}
                fontSize={bipolar ? 0.042 : 0.072}
                fill={lit || hot === i ? "var(--paper)" : "var(--ink)"}
                style={{ pointerEvents: "none" }}
              >
                {bipolar ? seg.ch_names[i].replace("-", "–") : seg.ch_names[i]}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
