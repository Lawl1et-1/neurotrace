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

All numbers are means over held-out test subjects ± the half-width of the 95 % t-interval, with n the
number of test subjects; event counts and hours are pooled within a subject first. Regenerate any table
with `python scripts/results_table.py <dataset>`. An earlier version of this evaluation had two flaws
(window-to-window connectivity change as an anomaly score, and early alarms counted as detections) and
an easy benchmark; both are fixed, see `docs/DECISIONS.md` D13–D15.

### Synthetic ground truth (v2: 20 subjects, 5-fold leave-subjects-out)

| Model | n | Window AUROC | AUPRC | Sensitivity | FA / h | Latency (s) | Edge AUROC | Event-edge AUROC | Path seed hit |
|---|---|---|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 20 | 0.909 ± 0.027 | 0.714 ± 0.051 | 0.17 ± 0.09 | 0.0 ± 0.0 | 2.6 ± 2.5 | 0.930 ± 0.021 | 0.854 ± 0.033 | 0.32 ± 0.12 |
| – diffusion GNN instead of GAT | 20 | 0.907 ± 0.029 | 0.711 ± 0.053 | 0.17 ± 0.09 | 0.0 ± 0.0 | 2.0 ± 1.6 | 0.933 ± 0.016 | 0.864 ± 0.034 | 0.37 ± 0.13 |
| – distance-prior graph only | 20 | 0.838 ± 0.066 | 0.646 ± 0.093 | 0.15 ± 0.09 | 0.0 ± 0.0 | 1.6 ± 1.1 | — | — | — |
| – no graph | 20 | 0.833 ± 0.064 | 0.631 ± 0.092 | 0.15 ± 0.09 | 0.0 ± 0.0 | 1.6 ± 1.1 | — | — | — |
| – no Transformer | 20 | 0.901 ± 0.029 | 0.699 ± 0.051 | 0.13 ± 0.08 | 0.0 ± 0.0 | 1.8 ± 2.9 | 0.926 ± 0.022 | 0.853 ± 0.036 | 0.27 ± 0.11 |
| – edge module without VAR evidence | 20 | 0.920 ± 0.023 | 0.747 ± 0.042 | 0.13 ± 0.09 | 0.0 ± 0.0 | 1.5 ± 1.7 | 0.829 ± 0.033 | 0.847 ± 0.038 | 0.40 ± 0.16 |
| – masked-window objective | 20 | 0.888 ± 0.039 | 0.683 ± 0.061 | 0.15 ± 0.09 | 0.0 ± 0.0 | 1.3 ± 1.1 | 0.917 ± 0.024 | 0.838 ± 0.033 | 0.28 ± 0.09 |
| – + supervised head (uses labels) | 20 | 0.894 ± 0.040 | 0.740 ± 0.053 | 0.75 ± 0.11 | 8.0 ± 8.3 | 3.6 ± 0.5 | 0.894 ± 0.034 | 0.798 ± 0.060 | 0.25 ± 0.14 |
| Per-channel CNN-LSTM | 20 | 0.873 ± 0.040 | 0.655 ± 0.070 | 0.15 ± 0.09 | 0.0 ± 0.0 | 1.2 ± 0.8 | — | — | — |
| Band-power deviation | 20 | 0.935 ± 0.021 | 0.792 ± 0.035 | 0.45 ± 0.15 | 1.8 ± 3.7 | 1.7 ± 0.7 | — | — | — |
| Pearson deviation | 20 | 0.825 ± 0.033 | 0.589 ± 0.054 | 0.45 ± 0.16 | 8.0 ± 7.8 | 3.1 ± 1.2 | 0.717 ± 0.022 | 0.605 ± 0.038 | 0.08 ± 0.07 |
| Coherence deviation | 20 | 0.814 ± 0.049 | 0.618 ± 0.057 | 0.47 ± 0.17 | 6.2 ± 5.6 | 2.9 ± 1.3 | 0.927 ± 0.011 | 0.812 ± 0.036 | 0.08 ± 0.07 |
| Granger (per window) | 20 | 0.789 ± 0.046 | 0.537 ± 0.062 | 0.33 ± 0.15 | 9.8 ± 6.9 | 2.6 ± 1.1 | 0.968 ± 0.011 | 0.868 ± 0.037 | 0.48 ± 0.15 |
| PDC (per window) | 20 | 0.751 ± 0.058 | 0.456 ± 0.076 | 0.28 ± 0.15 | 11.5 ± 5.6 | 2.5 ± 1.8 | 0.926 ± 0.017 | 0.832 ± 0.046 | 0.23 ± 0.13 |
| DTF (per window) | 20 | 0.746 ± 0.051 | 0.462 ± 0.082 | 0.32 ± 0.14 | 8.0 ± 5.7 | 3.2 ± 1.2 | 0.921 ± 0.014 | 0.815 ± 0.040 | 0.37 ± 0.13 |

What this says:
- **Edge recovery.** The learned per-window graph recovers the injected directed couplings with an edge
  AUROC of 0.93 ± 0.02; per-window conditional Granger causality, the natural estimator for a linear
  autoregressive world, does better (0.97 ± 0.01). Without the conditional VAR evidence the edge module
  falls to 0.83.
- **Detection.** The graph matters for this model: 0.91 ± 0.03 window AUROC with the learned graph vs
  0.83–0.84 with no graph or only the distance prior. A simple band-power deviation score is still at
  least as good (0.94 ± 0.02; the intervals overlap). Connectivity-deviation baselines reach 0.75–0.83.
- **Event metrics are per-detector operating points.** Thresholds are the 99th percentile of each
  detector's scores on a held-out subject's normal windows: the self-supervised models end up with no
  false alarms and low sensitivity (≈ 0.15), the connectivity baselines with higher sensitivity at
  6–12 false alarms per hour. Each test subject has only 4 minutes of data, so false-alarm rates are
  rough. Compare AUROC/AUPRC across rows, and sensitivity only together with its false-alarm rate.
- **Propagation paths.** Starting from the channel whose outgoing dependencies grew most, the learned
  graph finds the true seed in 0.32 ± 0.12 of events, Granger in 0.48 ± 0.15. The learned gates are
  bounded, so strong and weak new couplings look alike (D14). A rule that also weighed band-power rise
  looked better on the first benchmark and did not hold up on this one; it is kept in the CSVs, not used.

### CHB-MIT (6 subjects, leave-one-subject-out)

Each patient is held out once (validation on another patient, the rest for training). Mean ± 95 %
CI half-width over the 6 test patients; score components are z-scored against the same recording's
preceding 10 minutes (D12). Source: `results/chbmit/summary.csv`.

| Model | n | Window AUROC | AUPRC | Sensitivity | FA / h | Latency (s) |
|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 6 | 0.840 ± 0.172 | 0.273 ± 0.171 | 0.18 ± 0.18 | 1.0 ± 0.7 | 14.2 ± 11.0 |
| – distance-prior graph only | 6 | 0.727 ± 0.187 | 0.156 ± 0.156 | 0.39 ± 0.36 | 3.9 ± 3.3 | 21.6 ± 22.3 |
| – no graph | 6 | 0.721 ± 0.190 | 0.152 ± 0.138 | 0.39 ± 0.36 | 3.7 ± 3.3 | 22.3 ± 22.5 |
| – + supervised head (uses labels) | 6 | 0.970 ± 0.021 | 0.786 ± 0.160 | 0.90 ± 0.18 | 0.5 ± 0.7 | 14.0 ± 11.2 |
| Per-channel CNN-LSTM | 6 | 0.611 ± 0.158 | 0.088 ± 0.054 | 0.38 ± 0.37 | 3.5 ± 3.2 | 22.9 ± 22.3 |
| Band-power deviation | 6 | 0.944 ± 0.074 | 0.543 ± 0.160 | 0.70 ± 0.39 | 4.2 ± 1.6 | 12.1 ± 7.1 |
| Pearson deviation | 6 | 0.819 ± 0.113 | 0.168 ± 0.154 | 0.47 ± 0.37 | 3.7 ± 1.1 | 19.0 ± 9.9 |
| Granger (per window) | 6 | 0.853 ± 0.082 | 0.214 ± 0.197 | 0.41 ± 0.35 | 2.6 ± 0.5 | 13.3 ± 8.9 |

What this says:
- **The simplest detector wins.** Band-power deviation has the best window AUROC (0.94 ± 0.07) and
  AUPRC (0.54) of the label-free detectors and catches the most seizures (0.70), at ≈ 4 false alarms
  per hour.
- **The learned graph carries signal, but not enough.** The full self-supervised model reaches
  0.84 ± 0.17 window AUROC vs 0.72–0.73 for the same model with no graph or the distance prior only;
  the gap comes mostly from two patients (chb01: 0.97 vs 0.63, chb08: 0.72 vs 0.62), and chb02 is near
  chance for every self-supervised variant. At its threshold it is the quietest detector (1.0 false
  alarm per hour) and misses most seizures (sensitivity 0.18).
- **The supervised head** (0.97 ± 0.02, sensitivity 0.90) is trained on the other patients' seizure
  labels. It shows the representation is useful for detection; it is an upper reference, not a
  label-free competitor.
- **Six patients.** Intervals are wide and per-patient results vary a lot; these numbers say nothing
  about other populations, centres or the TUH corpus. Coherence, PDC, DTF and the remaining ablations
  were not run on CHB-MIT, to keep the run to a few GPU hours.

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
