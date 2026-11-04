# NEUROTRACE

> **Research prototype — not a medical device, not for clinical use.**
> Nothing in this repository may be used to diagnose, monitor or treat anyone.

**Live demo:** https://lawl1et-1.github.io/neurotrace/

NEUROTRACE learns how the relationships between EEG channels change over time, flags unusual
changes in those relationships, and outputs a dynamic, interpretable map of them.

- **Spatial:** a graph neural network (PyTorch Geometric GAT, or directed diffusion convolution) over the
  electrodes, on a graph that mixes a distance prior with a **learned, directed, per-window adjacency**.
- **Temporal:** a causal Transformer over the sequence of per-window graph embeddings, trained
  self-supervised to predict the next window.
- **Explainability:** the learned graph over time, attention maps, Integrated Gradients (Captum) over
  channel × window × frequency band, edge masking, and a **propagation path** for every flagged event.

Two rules hold everywhere in the code, the results and the site:
learned edges are **directed statistical dependencies, not proof of causation**, and electrodes are
**not brain regions** (volume conduction spreads one source over many electrodes).

## Contents

| Path | What |
|---|---|
| `src/neurotrace/data/` | MNE preprocessing, windowing, band power, synthetic generator, loaders (CHB-MIT, TUH, EEG-MMI), subject splits |
| `src/neurotrace/graph/` | 10-20 montage, distance prior, learned directed edge module |
| `src/neurotrace/models/` | node encoder, GAT / diffusion / no-graph spatial modules, temporal Transformer, full model |
| `src/neurotrace/baselines/` | Pearson, coherence, Granger, PDC, DTF; per-channel CNN-LSTM |
| `src/neurotrace/explain/` | propagation paths, Integrated Gradients, edge masking, attention |
| `src/neurotrace/eval/` | scoring, metrics (edge AUROC, AUROC/AUPRC, latency, false alarms/h), CV runner, figures |
| `src/neurotrace/app/` | FastAPI server for the visualizer |
| `web/` | React + three.js visualizer and project site (deployed to GitHub Pages) |
| `configs/` | Hydra configs (`experiment=smoke` for CPU, `full` synthetic, `chbmit`) |
| `results/` | CSVs and figures written by the evaluation |
| `docs/` | `DECISIONS.md`, `PAPER_OUTLINE.md` |

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -e ".[app,dev]"
make test
```

Everything runs on CPU with `experiment=smoke`; full training uses a GPU when one is available
(`device: auto`). A 6 GB GTX 1660 Ti is enough.

## Data

- **Synthetic (no download):** 19 damped oscillators on the 10-20 layout, a sparse network of lagged
  directed couplings, mild volume conduction, and injected events where a seed channel starts a slow
  rhythm and a tree of new couplings carries it outward. Every coupling and the true propagation order
  are known. `make data-synth` prints a summary.
- **CHB-MIT Scalp EEG** (PhysioNet, open access): `make data` downloads, for 6 subjects, every file
  with a seizure plus 3 seizure-free files each (~2.5 GB, ~61 h), then filters and windows them into
  `data/processed/chbmit/`. `python scripts/download_chbmit.py --all` gets everything (~43 GB).
- **TUH EEG Seizure Corpus:** needs registration; `neurotrace.data.datasets.tuh.load_tuh(root)` reads a
  local copy and re-references it to the same 18 bipolar channels.
- **EEG Motor Movement/Imagery:** `neurotrace.data.datasets.eegmmi.load_eegmmi([1, 2, 3])` (downloads via MNE).

## Train, evaluate, explain

```bash
make train CONFIG=full                    # one model on the first leave-subjects-out fold
make eval  CONFIG=full                    # synthetic: 4-fold CV, all ablations + baselines
make eval-chbmit                          # CHB-MIT: leave-one-subject-out (several GPU hours)
python scripts/export_explanations.py experiment=full     # JSON for the visualizer
python scripts/export_explanations.py experiment=chbmit
```

Ablations are config switches: `model.graph=none|static|learned`, `model.spatial=gat|diffusion`,
`model.temporal=transformer|none`, `model.var_order=0`, `train.objective=next|masked`,
`train.supervised=true`.

## Visualizer

```bash
make app        # builds web/ and serves http://localhost:8000/neurotrace/ with the JSON API
make app-dev    # hot-reload dev server
```

A 2D topomap and a 3D head with directed edges (width = strength, red = grew vs the previous minute),
EEG traces on a ruled chart with the anomaly score above them, a timeline of detected events, and for
each event its propagation path, top changed edges, edge-masking effects, Integrated Gradients and
attention. Synthetic recordings overlay the injected ground truth. Every state has its own link.

## Results

All numbers are means over held-out subjects (`results/*/summary.csv`).

**Synthetic (12 subjects, 4-fold leave-subjects-out)**

| Model | Window AUROC | AUPRC | Edge AUROC | Event-edge AUROC | Path seed hit |
|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 0.950 | 0.774 | 0.944 | 0.772 | 0.25 |
| · diffusion GNN | 0.945 | 0.776 | 0.940 | 0.765 | 0.25 |
| · no Transformer | 0.948 | 0.778 | 0.951 | 0.815 | 0.22 |
| · no VAR evidence in the edge module | 0.962 | 0.784 | 0.729 | 0.722 | 0.28 |
| · static distance graph | 0.944 | 0.785 | — | — | — |
| · no graph | 0.945 | 0.784 | — | — | — |
| · supervised head (uses labels) | 0.980 | 0.868 | 0.575 | 0.730 | 0.42 |
| Per-channel CNN-LSTM | 0.950 | 0.790 | — | — | — |
| Band-power deviation | **0.981** | 0.860 | — | — | — |
| Granger (per window) | 0.219* | 0.106 | **0.974** | **0.900** | **0.89** |
| PDC / DTF (per window) | 0.35 / 0.32* | 0.13 / 0.15 | 0.931 / 0.921 | 0.85 / 0.82 | 0.17 / 0.28 |

\* connectivity-*change* used as an anomaly score; the graph is stable during an event and changes at
its edges, so this score is below chance.

What this says, plainly:
- The learned graph recovers the injected directed couplings well (0.94), close to per-window
  conditional Granger (0.97), which is the right estimator for a linear autoregressive world. Without the
  conditional VAR evidence the edge module drops to 0.73.
- For anomaly detection the graph adds nothing measurable on this data: all variants are within noise of
  each other, and a simple band-power deviation score is better.
- Propagation paths from the learned graph find the true seed in only ~25 % of events; Granger-based
  paths find it in 89 %. Explanation quality lags edge recovery.

**CHB-MIT (6 subjects, leave-one-subject-out):** see `results/chbmit/summary.csv` and the live site.

## Limitations

- **Not a medical device.** Research and education only; no clinical validation.
- **Dependencies, not causes.** Shared inputs, filtering, the reference and volume conduction create
  directed edges without any causal pathway.
- **Electrodes, not regions.** CHB-MIT channels are bipolar derivations placed at their midpoints;
  source reconstruction is an optional extension with its own errors.
- **Window-resolution timing.** Paths are ordered at the 1 s stride; true delays are tens of
  milliseconds, so only path order is testable, and only on synthetic data.
- **Small real cohort.** Six paediatric patients from one centre, mostly seizure files; wide intervals,
  no claim about other populations or the TUH corpus.
- **Sensitive sparsity.** `lambda_sparse` 0.02–0.03 works, 0.05 collapses the graph; tuned on synthetic
  data and transferred unchanged.
- **Linear synthetic world.** It favours Granger-type estimators; it is a sanity check, not a benchmark.

See `docs/DECISIONS.md` for the reasoning behind each choice and `docs/PAPER_OUTLINE.md` for the
write-up plan.

## Data licence

CHB-MIT Scalp EEG Database: Shoeb, A. (2009), PhysioNet, Open Data Commons Attribution License v1.0.
Goldberger et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23), 2000.

Code: MIT licence.
