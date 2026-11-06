// Types and loaders for the JSON exported by scripts/export_explanations.py.

export interface SegmentMeta {
  id: string;
  dataset: "synthetic" | "chbmit";
  subject: string;
  record: string;
  duration_s: number;
  n_detected: number;
  n_labelled: number;
}

export type SummaryRow = Record<string, string | number | null>;

export interface Index {
  disclaimer: string;
  edge_semantics: string;
  bands: string[];
  segments: SegmentMeta[];
  summary: Record<string, SummaryRow[]>;
  /** dataset -> analysis CSV name (per_patient, wilcoxon, matched_fa, term_ablation, ...) -> rows */
  analysis?: Record<string, Record<string, SummaryRow[]>>;
}

export interface PathStep {
  ch: number;
  t: number;
  via?: number | null;
  w?: number;
  z?: number;
}

export interface EdgeDelta {
  src: number;
  dst: number;
  delta?: number;
  effect?: number;
}

export interface DetectedEvent {
  start: number;
  end: number;
  onset_s: number;
  end_s: number;
  peak_score: number;
  explained_window: number;
  graph_path: PathStep[];
  signal_path: PathStep[];
  top_edges: EdgeDelta[];
  edge_masking: EdgeDelta[];
  ig_bands: number[][]; // channel x band
  ig_channel_time: number[][]; // window x channel
  ig_delta: number;
  ig_rel_delta?: number;
  ig_steps?: number;
  ig_converged?: boolean;
  shared_share?: number; // share of the graph increase on derivations sharing an electrode (excluded)
  hub?: number | null;
  hub_share?: number;
  hub_warning?: boolean;
  attention: number[] | null;
}

export interface TruthEvent {
  onset_s: number;
  offset_s: number;
  seed: number;
  order: number[];
  arrival_s: number[];
  edges: [number, number][];
}

export interface Segment {
  id: string;
  dataset: string;
  subject: string;
  record: string;
  offset_s: number;
  ch_names: string[];
  pos2d: [number, number][];
  pos3d: [number, number, number][];
  stride_s: number;
  window_s: number;
  trace_hz: number;
  times: number[];
  score: number[];
  threshold: number;
  components: Record<string, number[]>;
  adj_scale: number;
  adj: number[][][] | null; // window x src x dst, 0..99
  traces: number[][]; // channel x sample, z * 25
  labels: { start_s: number; end_s: number }[];
  events: DetectedEvent[];
  truth: { base_edges: [number, number][]; events: TruthEvent[] } | null;
  shared_pairs?: number[][];
}

const base = import.meta.env.BASE_URL;

export async function loadIndex(): Promise<Index> {
  const r = await fetch(`${base}data/index.json`);
  if (!r.ok) throw new Error(`index.json: HTTP ${r.status}`);
  return r.json();
}

const cache = new Map<string, Promise<Segment>>();

export function loadSegment(id: string): Promise<Segment> {
  if (!cache.has(id)) {
    cache.set(
      id,
      fetch(`${base}data/seg/${id}.json`)
        .then((r) => {
          if (!r.ok) throw new Error(`${id}: HTTP ${r.status}`);
          return r.json();
        })
        .catch((e) => {
          cache.delete(id); // a failed load must not be cached: picking the recording again retries
          throw e;
        }),
    );
  }
  return cache.get(id)!;
}

/** Mean learned adjacency over windows [a, b). */
export function meanAdj(seg: Segment, a: number, b: number): number[][] {
  const c = seg.ch_names.length;
  const out = Array.from({ length: c }, () => new Array(c).fill(0));
  if (!seg.adj) return out;
  const lo = Math.max(0, a);
  const hi = Math.min(seg.adj.length, Math.max(b, lo + 1));
  for (let w = lo; w < hi; w++)
    for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) out[i][j] += seg.adj[w][i][j];
  const n = hi - lo;
  for (let i = 0; i < c; i++) for (let j = 0; j < c; j++) out[i][j] /= n * 99;
  return out;
}

export interface Edge {
  src: number;
  dst: number;
  w: number; // current strength 0..1
  change: number; // vs the preceding minute
}

/** Strongest edges at window t with their change against the previous 60 windows. */
/** Ordered pairs of bipolar derivations that share an electrode: coupled by construction, never drawn. */
export function sharedSet(seg: Segment): Set<string> {
  const s = new Set<string>();
  for (const [i, j] of seg.shared_pairs ?? []) {
    s.add(`${i}-${j}`);
    s.add(`${j}-${i}`);
  }
  return s;
}

export function edgesAt(seg: Segment, t: number, k = 20): Edge[] {
  if (!seg.adj) return [];
  const c = seg.ch_names.length;
  const shared = sharedSet(seg);
  const now = meanAdj(seg, t - 1, t + 2);
  // change is only meaningful once there is a baseline minute (at least 20 s) behind the pen
  const before = t >= 20 ? meanAdj(seg, t - 60, t - 5) : now;
  const all: Edge[] = [];
  for (let i = 0; i < c; i++)
    for (let j = 0; j < c; j++)
      if (i !== j && !shared.has(`${i}-${j}`)) all.push({ src: i, dst: j, w: now[i][j], change: now[i][j] - before[i][j] });
  all.sort((a, b) => b.w - a.w);
  const top = all[0]?.w ?? 0;
  // keep the strong edges only: a faint long-range edge is noise on a map
  return all.slice(0, k).filter((e) => e.w >= 0.4 * top);
}

export function eventAt(seg: Segment, t: number): number {
  return seg.events.findIndex((e) => t >= e.start - 3 && t < e.end + 3);
}

export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
};
