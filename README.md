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
preceding 10 minutes (D12). Every ablation and every classical baseline is run; connectivity is
estimated on every window. Thresholds are the 99.5th percentile of the validation patient's normal
windows, for the models and (since D16) for the baselines too. The two combined detectors have no
calibrated threshold, so only their window metrics and matched-FA operating points are given.
Sources: `results/chbmit/summary.csv` and the analysis files written by
`scripts/analyze_results.py` (D19).

| Model | n | Window AUROC | AUPRC | Sensitivity | FA / h | Latency (s) |
|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 6 | 0.838 ± 0.175 | 0.273 ± 0.170 | 0.18 ± 0.18 | 1.0 ± 0.7 | 14.2 ± 11.0 |
| – diffusion GNN instead of GAT | 6 | 0.835 ± 0.184 | 0.273 ± 0.174 | 0.18 ± 0.18 | 1.0 ± 0.7 | 14.4 ± 11.0 |
| – distance-prior graph only | 6 | 0.727 ± 0.187 | 0.156 ± 0.156 | 0.39 ± 0.36 | 3.9 ± 3.4 | 21.6 ± 22.3 |
| – no graph | 6 | 0.721 ± 0.190 | 0.152 ± 0.138 | 0.39 ± 0.36 | 3.7 ± 3.3 | 22.3 ± 22.5 |
| – no Transformer | 6 | 0.830 ± 0.176 | 0.267 ± 0.171 | 0.18 ± 0.18 | 1.0 ± 0.8 | 14.2 ± 11.0 |
| – edge module without VAR evidence | 6 | 0.855 ± 0.156 | 0.288 ± 0.180 | 0.18 ± 0.18 | 1.0 ± 0.7 | 13.9 ± 11.4 |
| – masked-window objective | 6 | 0.861 ± 0.165 | 0.292 ± 0.172 | 0.18 ± 0.18 | 1.0 ± 0.7 | 14.4 ± 11.0 |
| – + supervised head (uses labels) | 6 | 0.977 ± 0.016 | 0.817 ± 0.113 | 0.98 ± 0.06 | 0.6 ± 0.5 | 12.2 ± 6.4 |
| Per-channel CNN-LSTM | 6 | 0.611 ± 0.158 | 0.088 ± 0.054 | 0.38 ± 0.37 | 3.5 ± 3.2 | 22.9 ± 22.3 |
| Band-power deviation | 6 | 0.944 ± 0.074 | 0.543 ± 0.160 | 0.59 ± 0.49 | 2.5 ± 1.6 | 13.4 ± 13.3 |
| Pearson deviation | 6 | 0.795 ± 0.120 | 0.144 ± 0.142 | 0.40 ± 0.40 | 3.0 ± 1.4 | 16.4 ± 17.4 |
| Coherence deviation | 6 | 0.836 ± 0.094 | 0.185 ± 0.144 | 0.30 ± 0.30 | 2.2 ± 0.8 | 21.7 ± 27.5 |
| Granger (per window) | 6 | 0.852 ± 0.074 | 0.213 ± 0.196 | 0.42 ± 0.36 | 2.4 ± 0.9 | 18.8 ± 23.4 |
| PDC (per window) | 6 | 0.786 ± 0.139 | 0.143 ± 0.129 | 0.25 ± 0.29 | 1.7 ± 0.8 | 10.1 ± 14.3 |
| DTF (per window) | 6 | 0.788 ± 0.153 | 0.166 ± 0.151 | 0.25 ± 0.29 | 1.9 ± 0.8 | 9.6 ± 12.4 |
| Band power + learned-graph change | 6 | 0.920 ± 0.105 | 0.419 ± 0.131 | — | — | — |
| Band power + Granger change | 6 | 0.950 ± 0.053 | 0.502 ± 0.163 | — | — | — |

#### Per-patient window AUROC

| Patient | Full | No graph | Dist. graph | Superv. | Band power | Granger | BP + graph | BP + Granger |
|---|---|---|---|---|---|---|---|---|
| chb01 | 0.968 | 0.625 | 0.599 | 0.982 | 0.983 | 0.968 | 0.973 | 0.985 |
| chb02 | 0.551 | 0.461 | 0.535 | 0.977 | 0.809 | 0.782 | 0.720 | 0.859 |
| chb03 | 0.935 | 0.818 | 0.821 | 0.987 | 0.971 | 0.864 | 0.953 | 0.970 |
| chb05 | 0.887 | 0.888 | 0.913 | 0.996 | 0.978 | 0.773 | 0.955 | 0.968 |
| chb08 | 0.724 | 0.615 | 0.570 | 0.952 | 0.926 | 0.871 | 0.927 | 0.925 |
| chb10 | 0.963 | 0.919 | 0.922 | 0.967 | 0.995 | 0.856 | 0.989 | 0.991 |

#### Paired Wilcoxon signed-rank tests over patients (two-sided)

| Metric | A | B | n | median A − B | A better in | p |
|---|---|---|---|---|---|---|
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | – no graph | 6 | +0.100 | 5/6 | 0.0625 |
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | Band-power deviation | 6 | -0.063 | 0/6 | 0.0312 |
| AUROC | Band power + learned-graph change | Band-power deviation | 6 | -0.014 | 1/6 | 0.0625 |
| AUROC | Band power + learned-graph change | Band power + Granger change | 6 | -0.013 | 1/6 | 0.0625 |
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | Granger (per window) | 6 | +0.036 | 4/6 | 1 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | – no graph | 6 | +0.058 | 5/6 | 0.219 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | Band-power deviation | 6 | -0.250 | 0/6 | 0.0312 |
| AUPRC | Band power + learned-graph change | Band-power deviation | 6 | -0.142 | 0/6 | 0.0312 |
| AUPRC | Band power + learned-graph change | Band power + Granger change | 6 | -0.071 | 0/6 | 0.0312 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | Granger (per window) | 6 | +0.079 | 3/6 | 0.688 |

#### Sensitivity (latency, s) at matched false-alarm rates; threshold set per test patient (oracle)

| Detector | ≤ 0.5 FA/h | ≤ 1 FA/h | ≤ 2 FA/h |
|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 0.25 ± 0.24 (16 s) | 0.30 ± 0.28 (14 s) | 0.41 ± 0.41 (12 s) |
| – diffusion GNN instead of GAT | 0.25 ± 0.24 (16 s) | 0.30 ± 0.28 (14 s) | 0.41 ± 0.41 (11 s) |
| – distance-prior graph only | 0.22 ± 0.27 (23 s) | 0.24 ± 0.29 (22 s) | 0.30 ± 0.37 (18 s) |
| – no graph | 0.15 ± 0.19 (12 s) | 0.24 ± 0.29 (23 s) | 0.27 ± 0.31 (22 s) |
| – no Transformer | 0.25 ± 0.24 (16 s) | 0.30 ± 0.28 (14 s) | 0.41 ± 0.41 (12 s) |
| – edge module without VAR evidence | 0.25 ± 0.24 (15 s) | 0.30 ± 0.29 (13 s) | 0.37 ± 0.38 (8 s) |
| – masked-window objective | 0.25 ± 0.24 (16 s) | 0.32 ± 0.31 (13 s) | 0.44 ± 0.44 (10 s) |
| – + supervised head (uses labels) | 0.98 ± 0.06 (13 s) | 0.98 ± 0.06 (9 s) | 1.00 ± 0.00 (6 s) |
| Per-channel CNN-LSTM | 0.15 ± 0.19 (14 s) | 0.21 ± 0.24 (19 s) | 0.21 ± 0.24 (18 s) |
| Band-power deviation | 0.27 ± 0.19 (17 s) | 0.59 ± 0.38 (13 s) | 0.73 ± 0.39 (13 s) |
| Pearson deviation | 0.23 ± 0.31 (20 s) | 0.28 ± 0.41 (20 s) | 0.31 ± 0.43 (18 s) |
| Coherence deviation | 0.02 ± 0.06 (9 s) | 0.19 ± 0.35 (17 s) | 0.24 ± 0.35 (15 s) |
| Granger (per window) | 0.22 ± 0.28 (12 s) | 0.25 ± 0.33 (11 s) | 0.40 ± 0.40 (20 s) |
| PDC (per window) | 0.05 ± 0.08 (24 s) | 0.17 ± 0.24 (15 s) | 0.26 ± 0.38 (9 s) |
| DTF (per window) | 0.05 ± 0.08 (23 s) | 0.19 ± 0.29 (16 s) | 0.26 ± 0.36 (13 s) |
| Band power + learned-graph change | 0.17 ± 0.17 (19 s) | 0.40 ± 0.37 (14 s) | 0.61 ± 0.39 (10 s) |
| Band power + Granger change | 0.28 ± 0.34 (16 s) | 0.49 ± 0.34 (14 s) | 0.64 ± 0.41 (11 s) |

#### Score terms: window AUROC of each term alone, and its share of the score at alarm windows

| Variant | AUROC pred. error | AUROC graph change | AUROC forecast error | share pred. | share graph | share forecast |
|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 0.702 | 0.513 | 0.844 | 0.21 | 0.03 | 0.76 |
| – diffusion GNN instead of GAT | 0.701 | 0.509 | 0.844 | 0.22 | 0.02 | 0.76 |
| – distance-prior graph only | 0.727 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |
| – no graph | 0.721 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |
| – no Transformer | 0.661 | 0.509 | 0.844 | 0.21 | 0.03 | 0.76 |
| – edge module without VAR evidence | 0.706 | 0.633 | 0.845 | 0.19 | 0.03 | 0.78 |
| – masked-window objective | 0.807 | 0.514 | 0.843 | 0.21 | 0.02 | 0.76 |
| Per-channel CNN-LSTM | 0.611 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |

What this says:
- **The simplest detector wins.** Band-power deviation has the best window AUROC (0.94 ± 0.07) and
  AUPRC (0.54) of the label-free detectors. It beats the full model in all 6 patients (paired
  Wilcoxon p = 0.031, the smallest p a two-sided test can give with 6 patients). At a matched
  1 false alarm per hour it detects 0.59 of seizures against 0.30 for the full model.
- **The learned graph carries signal, weakly.** With the learned graph the model reaches
  0.84 ± 0.18; with no graph or the distance prior only, 0.72–0.73. The full model is better in
  5 of 6 patients, but p = 0.06, and most of the gap comes from chb01 (0.97 vs 0.63).
- **Adding the learned graph to band power does not help.** Band power + learned-graph change
  (0.92) is below band power alone in 5 of 6 patients, and below band power + Granger change (0.95).
  The learned-graph-change term on its own has an AUROC of about 0.51: it does not separate
  seizures from normal windows.
- **Why the ablations alarm alike.** Each variant has its own scores and its own threshold. Their
  alarms agree because ~76 % of the score at alarm windows comes from the VAR forecast error,
  which the GAT, Transformer and objective ablations leave unchanged. The prediction error
  contributes ~21 % and the graph change ~3 %. Alarm windows overlap 87–97 % (Jaccard,
  `alarm_overlap.csv`) between the learned-graph variants and ~21 % with the no-graph model.
  "No VAR evidence" removes the VAR fit from the edge module's input, not the forecast-error term
  from the score. So all learned-graph variants share one operating point: 1.0 false alarm per hour
  at sensitivity 0.18.
- **The supervised head** (0.98 ± 0.02, sensitivity 0.98) is trained on the other patients' seizure
  labels. It shows the representation is useful for detection; it is an upper reference, not a
  label-free competitor.
- **Six patients.** Intervals are wide and per-patient results vary a lot. The matched-FA thresholds
  are set on each test patient (an oracle) and only compare detectors at the same false-alarm
  budget. The all-patient run (chb01–chb24, chb21 counted with chb01) is in progress.

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
