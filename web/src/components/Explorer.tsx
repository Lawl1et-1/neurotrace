import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { edgesAt, eventAt, fmtTime, loadSegment, type Index, type Segment } from "../data";
import { Arrow, Code, Pause, Play, Step } from "./Icons";
import MontageMap from "./MontageMap";
import Traces from "./Traces";
import EventReport from "./EventReport";

const Head3D = lazy(() => import("./Head3D"));
const REPO = "https://github.com/Lawl1et-1/neurotrace";
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { seg: p.get("seg"), t: p.get("t") ? Number(p.get("t")) : null, ev: p.get("ev") ? Number(p.get("ev")) : null };
}

export default function Explorer({ index }: { index: Index }) {
  const initial = useRef(readHash());
  const defaultSeg = index.segments.find((s) => s.id === initial.current.seg)?.id ?? [...index.segments].sort((a, b) => b.n_detected - a.n_detected || (a.dataset === "synthetic" ? -1 : 1))[0]?.id;
  const [segId, setSegId] = useState<string>(defaultSeg);
  const [seg, setSeg] = useState<Segment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<"2d" | "3d">("2d");
  const [hot, setHot] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [reveal, setReveal] = useState(0);
  const [showTruth, setShowTruth] = useState(true);
  const [showChange, setShowChange] = useState(true);
  const [span, setSpan] = useState(24);

  useEffect(() => {
    if (!segId) return;
    setError(null);
    setSeg(null);
    loadSegment(segId)
      .then((s) => {
        setSeg(s);
        const h = initial.current;
        if (h.seg === segId && h.ev != null && s.events[h.ev]) {
          setSelected(h.ev);
          setT(s.events[h.ev].start);
        } else if (h.seg === segId && h.t != null) {
          setT(Math.min(h.t, s.times.length - 1));
          setSelected(null);
        } else {
          // open a little before the first detected event so the reviewer watches it arrive
          setT(s.events.length ? Math.max(0, s.events[0].start - 14) : 0);
          setSelected(null);
          setPlaying(!reduced());
        }
      })
      .catch((e) => setError(String(e)));
  }, [segId]);

  // follow links pasted or clicked while the page is open
  useEffect(() => {
    const onHash = () => {
      const h = readHash();
      if (h.seg && index.segments.some((s) => s.id === h.seg)) {
        initial.current = h;
        setPlaying(false);
        if (h.seg === segId) {
          setSegId("");
          setTimeout(() => setSegId(h.seg!), 0);
        } else setSegId(h.seg);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [index, segId]);

  useEffect(() => {
    const w = () => setSpan(window.innerWidth < 640 ? 10 : window.innerWidth < 1100 ? 16 : 24);
    w();
    window.addEventListener("resize", w);
    return () => window.removeEventListener("resize", w);
  }, []);

  // the chart feed: four windows per second while playing
  useEffect(() => {
    if (!playing || !seg) return;
    const id = window.setInterval(() => setT((v) => (v + 1 < seg.times.length ? v + 1 : (setPlaying(false), v))), 250);
    return () => window.clearInterval(id);
  }, [playing, seg]);

  // while playing, the event under the pen becomes the selected one
  useEffect(() => {
    if (!seg || !playing) return;
    const e = eventAt(seg, t);
    if (e >= 0 && e !== selected) setSelected(e);
  }, [t, seg, playing, selected]);

  // sequenced propagation reveal: one channel at a time along the path
  useEffect(() => {
    if (selected == null || !seg) return setReveal(0);
    const n = seg.events[selected].graph_path.length;
    if (reduced()) return setReveal(n);
    setReveal(1);
    let k = 1;
    const id = window.setInterval(() => {
      k += 1;
      setReveal(Math.min(k, n));
      if (k >= n) window.clearInterval(id);
    }, 420);
    return () => window.clearInterval(id);
  }, [selected, seg]);

  // every state has a link
  useEffect(() => {
    if (!seg) return;
    const p = new URLSearchParams({ seg: seg.id });
    if (selected != null) p.set("ev", String(selected));
    else p.set("t", String(t));
    history.replaceState(null, "", `#${p.toString()}`);
  }, [seg, t, selected]);

  const edges = useMemo(() => (seg ? edgesAt(seg, t) : []), [seg, t]);
  const ev = seg && selected != null ? seg.events[selected] : null;
  const litPath = useMemo(() => (ev ? ev.graph_path.slice(0, reveal) : []), [ev, reveal]);
  const litSet = useMemo(() => new Set(litPath.map((p) => p.ch)), [litPath]);
  const truth = useMemo(() => {
    if (!seg?.truth) return null;
    const now = seg.times[t] ?? 0;
    return seg.truth.events.find((e) => now + seg.window_s >= e.onset_s && now <= e.offset_s) ?? null;
  }, [seg, t]);

  const selectEvent = useCallback(
    (i: number) => {
      if (!seg) return;
      setPlaying(false);
      setSelected(i);
      setT(Math.min(seg.times.length - 1, seg.events[i].start + 2));
      document.getElementById("report")?.scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "start" });
    },
    [seg],
  );

  const segMeta = index.segments.find((s) => s.id === segId);
  const isSyn = seg?.dataset === "synthetic";
  const now = seg ? seg.times[t] ?? 0 : 0;

  return (
    <>
      <section className="explorer ruled" id="record" aria-label="Interactive record">
        <div className="gutter-top">
          <span className="code">REC</span>
          <span className="tnum">{seg ? fmtTime(now) : "--:--"}</span>
        </div>

        <div className="map-pane">
          <div className="map-tools">
            <label>
              <span className="sr-only">Recording</span>
              <select className="seg-select" value={segId} onChange={(e) => { initial.current = { seg: null, t: null, ev: null }; setPlaying(false); setSegId(e.target.value); }}>
                {(["synthetic", "chbmit"] as const).map((ds) => {
                  const list = index.segments.filter((s) => s.dataset === ds);
                  if (!list.length) return null;
                  return (
                    <optgroup key={ds} label={ds === "synthetic" ? "Synthetic (known ground truth)" : "CHB-MIT scalp EEG (seizure files)"}>
                      {list.map((s) => (
                        <option key={s.id} value={s.id}>
                          {ds === "synthetic" ? `${s.subject} · ${s.n_labelled} injected events` : `${s.record} · ${s.n_labelled} annotated seizure${s.n_labelled === 1 ? "" : "s"}`}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </select>
            </label>
            <div className="toggle" role="group" aria-label="Map view">
              <button aria-pressed={view === "2d"} onClick={() => setView("2d")}>Topomap</button>
              <button aria-pressed={view === "3d"} onClick={() => setView("3d")}>3D head</button>
            </div>
            <div className="legend" role="group" aria-label="Map layers">
              <button aria-pressed={showChange} onClick={() => setShowChange((v) => !v)} title="Highlight edges that grew compared with the previous minute">
                <span className="swatch-line" style={{ borderColor: "var(--blue)" }} /> dependency
                <span className="swatch-line" style={{ borderColor: "var(--red)", marginLeft: 6 }} /> grew
              </button>
              {isSyn && (
                <button aria-pressed={showTruth} onClick={() => setShowTruth((v) => !v)} title="Show the couplings that were actually injected">
                  <span className="swatch-line" style={{ borderColor: "var(--green)", borderTopStyle: "dashed" }} /> ground truth
                </button>
              )}
            </div>
          </div>
          {seg && (
              <p className="map-note">
                {isSyn
                  ? "Synthetic subject held out from training: the dashed green arrows are the couplings that were really injected."
                  : `CHB-MIT ${seg.record}, held out from training (leave-one-subject-out). Green band: the clinicians' seizure annotation.`}{" "}
                {segMeta && `${seg.events.length} flagged, ${segMeta.n_labelled} labelled.`}
              </p>
            )}
          <div className="map-stage">
            {error && <p className="empty">Could not load this recording ({error}).</p>}
            {!seg && !error && <p className="empty">Loading the record…</p>}
            {seg && view === "2d" && (
              <MontageMap seg={seg} edges={edges} litPath={litPath} hot={hot} setHot={setHot} truth={truth} showTruth={isSyn && showTruth} showChange={showChange} />
            )}
            {seg && view === "3d" && (
              <Suspense fallback={<p className="empty">Loading the 3D head…</p>}>
                <Head3D seg={seg} edges={edges} litPath={litPath} hot={hot} />
              </Suspense>
            )}
          </div>
        </div>

        <div className="thesis">
          <h1>
            How EEG channels come to <em>predict each other</em>, and when that changes.
          </h1>
          <p className="reading">
            NEUROTRACE learns a <strong>directed graph between electrodes for every second</strong> of a recording,
            reads the sequence with a Transformer, and flags the moments the relationships change. Each flag comes with
            the path the change took. Arrows are <strong>directed statistical dependencies, not proof of causation</strong>.
          </p>
          <div className="actions">
            {seg && seg.events.length > 0 && (
              <button className="btn" onClick={() => selectEvent(selected ?? 0)}>
                Explain the {selected != null ? "selected" : "first"} event <Arrow />
              </button>
            )}
            <a className="btn ghost" href={REPO}>
              <Code /> Code &amp; results
            </a>
          </div>
        </div>

        {seg && (
          <Traces seg={seg} t={t} span={span} hot={hot} setHot={setHot} lit={litSet} selected={ev} onSeek={(v) => { setPlaying(false); setT(v); }} showTruth={isSyn && showTruth} />
        )}
        {seg && (
          <div className="transport">
            <div className="play">
              <button className="icon-btn" aria-label="Back 10 seconds" onClick={() => setT((v) => Math.max(0, v - 10))}><Step back /></button>
              <button className="icon-btn" aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying((p) => !p)}>{playing ? <Pause /> : <Play />}</button>
            </div>
            <Timeline seg={seg} t={t} setT={(v) => { setPlaying(false); setT(v); }} selected={selected} onEvent={selectEvent} />
          </div>
        )}
      </section>

      <section className="page ruled" id="report" aria-label="Event report">
        <div className="page-gutter">
          <span className="code">EVT</span>
          <span className="time tnum">{ev ? fmtTime(ev.onset_s) : "—"}</span>
        </div>
        <div className="page-body">
          {seg && <EventReport seg={seg} index={index} selected={selected} onSelect={selectEvent} reveal={reveal} setHot={setHot} />}
        </div>
      </section>
    </>
  );
}

function Timeline({ seg, t, setT, selected, onEvent }: { seg: Segment; t: number; setT: (v: number) => void; selected: number | null; onEvent: (i: number) => void }) {
  const n = seg.times.length;
  const W = 1000;
  const H = 56;
  const max = Math.max(seg.threshold * 1.5, ...seg.score);
  const min = Math.min(0, ...seg.score);
  const y = (v: number) => 30 - 26 * ((v - min) / (max - min || 1));
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const step = Math.max(1, Math.floor(n / 600));
  let d = "";
  for (let i = 0; i < n; i += step) d += `${i ? "L" : "M"}${x(i).toFixed(1)},${y(seg.score[i]).toFixed(1)}`;
  const dur = seg.times[n - 1] + seg.window_s;
  return (
    <div className="timeline">
      <input type="range" min={0} max={n - 1} value={t} onChange={(e) => setT(Number(e.target.value))} aria-label="Position in the recording" aria-valuetext={fmtTime(seg.times[t] ?? 0)} />
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        {seg.labels.map((l, i) => (
          <rect key={i} x={(l.start_s / dur) * W} width={Math.max(2, ((l.end_s - l.start_s) / dur) * W)} y={0} height={34} fill="var(--green)" opacity={0.16} />
        ))}
        <line x1={0} x2={W} y1={y(seg.threshold)} y2={y(seg.threshold)} stroke="var(--blue)" strokeDasharray="4 4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        <path d={d} fill="none" stroke="var(--blue)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        <line className="head" x1={x(t)} x2={x(t)} y1={0} y2={H} stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      </svg>
      {seg.events.map((e, i) => (
        <button
          key={i}
          onClick={() => onEvent(i)}
          aria-label={`Event ${i + 1} at ${fmtTime(e.onset_s)}`}
          aria-pressed={selected === i}
          style={{
            position: "absolute", left: `calc(${(e.start / Math.max(1, n - 1)) * 100}% - 9px)`, bottom: 2,
            width: 18, height: 18, borderRadius: "50%", cursor: "pointer", padding: 0,
            border: "2px solid var(--red)", background: selected === i ? "var(--red)" : "var(--paper)",
            font: "700 11px var(--f-label)", color: selected === i ? "#fff" : "var(--red)", zIndex: 2,
          }}
        >
          {i + 1}
        </button>
      ))}
    </div>
  );
}
