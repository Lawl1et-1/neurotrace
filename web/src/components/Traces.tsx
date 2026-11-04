import { useEffect, useRef, useState } from "react";
import type { DetectedEvent, Segment } from "../data";

interface Props {
  seg: Segment;
  t: number; // current window index (the pen position)
  span: number; // seconds visible
  hot: number | null;
  setHot: (ch: number | null) => void;
  lit: Set<number>;
  onSeek: (t: number) => void;
  showTruth: boolean;
  selected: DetectedEvent | null;
}

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const PEN = 0.72; // pen position as a fraction of the plot width: the paper feeds right to left

/** The record strip: anomaly score on top, then one trace per channel on ruled chart paper. */
export default function Traces({ seg, t, span, hot, setHot, lit, onSeek, showTruth, selected }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 800, h: 320 });
  const nCh = seg.ch_names.length;
  const rows = nCh + 1.6; // score row is taller
  const top = 14; // time ticks run along the top rule
  const rowH = (size.h - top) / rows;

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(wrap.current!);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvas.current!;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(size.w * dpr);
    cv.height = Math.round(size.h * dpr);
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w, h } = size;
    const ink = css("--ink");
    const blue = css("--blue");
    const red = css("--red");
    const green = css("--green");
    const minor = css("--grid-minor");
    const major = css("--grid-major");
    const pxPerS = w / span;
    const now = seg.times[Math.min(t, seg.times.length - 1)] + seg.window_s / 2;
    const t0 = now - PEN * span;
    const xOf = (s: number) => (s - t0) * pxPerS;

    // paper: minor rule every 0.2 s, major every second, scrolling with the record
    g.fillStyle = css("--paper");
    g.fillRect(0, 0, w, h);
    const first = Math.floor(t0 / 0.2) * 0.2;
    for (let s = first; s < t0 + span; s += 0.2) {
      const x = Math.round(xOf(s)) + 0.5;
      const isMajor = Math.abs(s - Math.round(s)) < 1e-6;
      g.strokeStyle = isMajor ? major : minor;
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, h);
      g.stroke();
    }
    for (let r = 0; r <= rows * 2; r++) {
      const y = Math.round(top + (r * rowH) / 2) + 0.5;
      g.strokeStyle = r % 2 === 0 ? major : minor;
      g.beginPath();
      g.moveTo(0, y);
      g.lineTo(w, y);
      g.stroke();
    }

    // labelled intervals: true events (dashed green, synthetic) and seizure annotations (CHB-MIT)
    for (const l of seg.labels) {
      const x1 = xOf(l.start_s);
      const x2 = xOf(l.end_s);
      if (x2 < 0 || x1 > w) continue;
      g.fillStyle = "rgba(47,143,106,0.07)";
      g.fillRect(x1, top + rowH * 1.6, x2 - x1, h);
      if (showTruth) {
        g.strokeStyle = green;
        g.setLineDash([6, 4]);
        g.lineWidth = 1.5;
        g.strokeRect(x1, top + rowH * 1.6 + 1, x2 - x1, h - top - rowH * 1.6 - 2);
        g.setLineDash([]);
      }
    }
    // detected events: red brackets on the score row; the selected event's bracket runs down every trace
    for (const e of seg.events) {
      const x1 = xOf(e.onset_s);
      const x2 = xOf(e.end_s);
      if (x2 < 0 || x1 > w) continue;
      const isSel = selected != null && selected.start === e.start;
      const yEnd = isSel ? h : top + 8;
      g.strokeStyle = red;
      g.lineWidth = isSel ? 2 : 1.5;
      g.beginPath();
      g.moveTo(x1, yEnd);
      g.lineTo(x1, top + 2);
      g.lineTo(x2, top + 2);
      g.lineTo(x2, yEnd);
      g.stroke();
      g.fillStyle = isSel ? "rgba(210,59,42,0.08)" : "rgba(210,59,42,0.04)";
      g.fillRect(x1, top, x2 - x1, h - top);
    }

    // anomaly score (one value per window, plotted at the window centre)
    const sTop = top + 5;
    const sH = rowH * 1.6 - 9;
    const tf = (v: number) => Math.sign(v) * Math.log1p(Math.abs(v)); // compressed: peaks can be 50x the threshold
    const win = seg.score.slice(Math.max(0, t - 200), t + 200).map(tf);
    const sMax = Math.max(tf(seg.threshold) * 1.8, ...win);
    const sMin = Math.min(0, ...win);
    const yS = (v: number) => sTop + sH * (1 - (tf(v) - sMin) / (sMax - sMin || 1));
    g.strokeStyle = blue;
    g.setLineDash([3, 3]);
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, yS(seg.threshold));
    g.lineTo(w, yS(seg.threshold));
    g.stroke();
    g.setLineDash([]);
    g.lineWidth = 2;
    g.lineJoin = "round";
    g.beginPath();
    let started = false;
    seg.times.forEach((tm, i) => {
      const x = xOf(tm + seg.window_s / 2);
      if (x < -10 || x > w + 10) return;
      const y = yS(seg.score[i]);
      if (!started) {
        g.moveTo(x, y);
        started = true;
      } else g.lineTo(x, y);
    });
    g.stroke();

    // channel traces in pen ink; channels on a revealed path in red; hovered channel heavier
    const hz = seg.trace_hz;
    const iA = Math.max(0, Math.floor(t0 * hz));
    const iB = Math.min(seg.traces[0].length, Math.ceil((t0 + span) * hz));
    const gain = rowH / 2 / 95;
    for (let c = 0; c < nCh; c++) {
      const yc = top + rowH * 1.6 + rowH * (c + 0.5);
      const tr = seg.traces[c];
      const isLit = lit.has(c);
      g.strokeStyle = isLit ? red : ink;
      g.globalAlpha = hot == null || hot === c || isLit ? 1 : 0.35;
      g.lineWidth = hot === c ? 1.6 : isLit ? 1.3 : 0.9;
      g.beginPath();
      for (let i = iA; i < iB; i++) {
        const x = xOf(i / hz);
        const y = yc - tr[i] * gain;
        if (i === iA) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      g.stroke();
    }
    g.globalAlpha = 1;

    // the pen line and time ticks (every 5 s)
    const px = Math.round(PEN * w) + 0.5;
    g.strokeStyle = ink;
    g.lineWidth = 1.25;
    g.beginPath();
    g.moveTo(px, 0);
    g.lineTo(px, h);
    g.stroke();
    g.fillStyle = css("--ink-3");
    g.font = `500 12px ${css("--f-label")}`;
    for (let s = Math.ceil(t0 / 5) * 5; s < t0 + span; s += 5) {
      const m = Math.floor(s / 60);
      g.fillText(`${m}:${String(Math.round(s % 60)).padStart(2, "0")}`, xOf(s) + 3, 11);
    }
  }, [seg, t, span, size, hot, lit, showTruth, selected, rowH, rows, nCh]);

  const seek = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    const now = seg.times[Math.min(t, seg.times.length - 1)] + seg.window_s / 2;
    const s = now - PEN * span + ((clientX - r.left) / r.width) * span;
    let best = 0;
    seg.times.forEach((tm, i) => {
      if (Math.abs(tm + seg.window_s / 2 - s) < Math.abs(seg.times[best] + seg.window_s / 2 - s)) best = i;
    });
    onSeek(best);
  };

  return (
    <div className="chart-strip">
      <div className="names" aria-hidden="false">
        <span className="score-name" style={{ position: "absolute", right: 8, top: top + (rowH * 1.6) / 2, transform: "translateY(-50%)", fontFamily: "var(--f-label)" }}>
          SCORE
        </span>
        {seg.ch_names.map((n, c) => (
          <button
            key={n}
            className={hot === c || lit.has(c) ? "hot" : ""}
            style={{ top: top + rowH * 1.6 + rowH * (c + 0.5) }}
            onMouseEnter={() => setHot(c)}
            onMouseLeave={() => setHot(null)}
            onFocus={() => setHot(c)}
            onBlur={() => setHot(null)}
          >
            {n}
          </button>
        ))}
      </div>
      <div
        className="plot"
        ref={wrap}
        style={{ height: Math.max(240, nCh * 11 + 46) }}
        onClick={(e) => seek(e.clientX)}
        title="Click to move the pen to this time"
      >
        <canvas ref={canvas} aria-label={`EEG traces of ${nCh} channels with the anomaly score above them`} role="img" />
      </div>
    </div>
  );
}
