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
- **CHB-MIT Scalp EEG** (PhysioNet, open access): `make data` downloads, for all 24 recordings (23 patients), every file
  with a seizure plus 3 seizure-free files each (~14 GB, ~294 h), then filters and windows them into
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

### CHB-MIT (23 patients, leave-one-patient-out)

Every patient is held out once (validation on another patient, the rest for training); chb21 is a
later recording of chb01 and is pooled with it. Each patient contributes every seizure file plus 3
seizure-free files (207 recordings, ~294 h). Mean ± 95 % CI half-width over the 23 test
patients; score components are z-scored against the same recording's preceding 10 minutes (D12).
Every ablation and every classical baseline is run; connectivity is estimated on every window.
Thresholds are the 99.5th percentile of the validation patient's normal windows, for the models and
(D16) for the baselines. The combined and score-term detectors have no calibrated threshold, so only
their window metrics and matched-FA operating points are given. Sources: `results/chbmit/summary.csv`
and the analysis files of `scripts/analyze_results.py` (D19, D20).

| Model | n | Window AUROC | AUPRC | Sensitivity | FA / h | Latency (s) |
|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 23 | 0.800 ± 0.059 | 0.225 ± 0.090 | 0.15 ± 0.11 | 0.3 ± 0.2 | 19.0 ± 8.3 |
| – diffusion GNN instead of GAT | 23 | 0.798 ± 0.059 | 0.223 ± 0.090 | 0.14 ± 0.11 | 0.3 ± 0.2 | 18.8 ± 8.3 |
| – distance-prior graph only | 23 | 0.648 ± 0.077 | 0.090 ± 0.035 | 0.14 ± 0.11 | 1.0 ± 0.5 | 17.0 ± 9.4 |
| – no graph | 23 | 0.648 ± 0.077 | 0.095 ± 0.037 | 0.14 ± 0.11 | 1.1 ± 0.6 | 16.4 ± 9.1 |
| – no Transformer | 23 | 0.797 ± 0.059 | 0.222 ± 0.090 | 0.15 ± 0.11 | 0.3 ± 0.2 | 18.9 ± 8.2 |
| – edge module without VAR evidence | 23 | 0.826 ± 0.052 | 0.238 ± 0.093 | 0.15 ± 0.11 | 0.4 ± 0.2 | 17.8 ± 8.6 |
| – masked-window objective | 23 | 0.823 ± 0.056 | 0.239 ± 0.091 | 0.14 ± 0.11 | 0.3 ± 0.2 | 18.8 ± 8.3 |
| – + supervised head (uses labels) | 23 | 0.893 ± 0.061 | 0.585 ± 0.139 | 0.74 ± 0.16 | 4.6 ± 3.1 | 10.5 ± 4.0 |
| Per-channel CNN-LSTM | 23 | 0.592 ± 0.073 | 0.073 ± 0.029 | 0.11 ± 0.09 | 0.9 ± 0.5 | 15.6 ± 12.4 |
| Band-power deviation | 23 | 0.896 ± 0.049 | 0.442 ± 0.112 | 0.50 ± 0.20 | 3.0 ± 1.3 | 11.3 ± 3.8 |
| Pearson deviation | 23 | 0.762 ± 0.049 | 0.142 ± 0.060 | 0.35 ± 0.14 | 3.2 ± 0.9 | 14.0 ± 4.5 |
| Coherence deviation | 23 | 0.770 ± 0.051 | 0.135 ± 0.059 | 0.21 ± 0.13 | 2.6 ± 0.7 | 18.3 ± 9.6 |
| Granger (per window) | 23 | 0.808 ± 0.040 | 0.169 ± 0.072 | 0.31 ± 0.15 | 2.6 ± 0.8 | 19.5 ± 9.1 |
| PDC (per window) | 23 | 0.736 ± 0.058 | 0.117 ± 0.054 | 0.21 ± 0.13 | 2.3 ± 0.7 | 14.2 ± 6.8 |
| DTF (per window) | 23 | 0.753 ± 0.051 | 0.135 ± 0.060 | 0.25 ± 0.13 | 2.2 ± 0.5 | 17.0 ± 6.9 |
| Band power + learned-graph change | 23 | 0.901 ± 0.036 | 0.361 ± 0.096 | — | — | — |
| Band power + Granger change | 23 | 0.910 ± 0.036 | 0.408 ± 0.108 | — | — | — |

#### Per-patient window AUROC

| Patient | Full | No graph | Dist. graph | Superv. | Band power | Granger | BP + graph | BP + Granger |
|---|---|---|---|---|---|---|---|---|
| chb01 | 0.821 | 0.506 | 0.526 | 0.981 | 0.980 | 0.852 | 0.956 | 0.967 |
| chb02 | 0.468 | 0.357 | 0.335 | 0.965 | 0.809 | 0.782 | 0.727 | 0.859 |
| chb03 | 0.934 | 0.731 | 0.744 | 0.992 | 0.971 | 0.864 | 0.952 | 0.970 |
| chb04 | 0.760 | 0.381 | 0.393 | 0.895 | 0.784 | 0.726 | 0.896 | 0.829 |
| chb05 | 0.851 | 0.734 | 0.727 | 0.986 | 0.978 | 0.773 | 0.950 | 0.968 |
| chb06 | 0.795 | 0.489 | 0.492 | 0.781 | 0.631 | 0.870 | 0.845 | 0.836 |
| chb07 | 0.792 | 0.611 | 0.586 | 0.988 | 0.973 | 0.748 | 0.942 | 0.974 |
| chb08 | 0.749 | 0.605 | 0.578 | 0.979 | 0.926 | 0.871 | 0.926 | 0.925 |
| chb09 | 0.968 | 0.925 | 0.876 | 0.990 | 0.996 | 0.829 | 0.991 | 0.993 |
| chb10 | 0.939 | 0.819 | 0.835 | 0.932 | 0.995 | 0.856 | 0.988 | 0.991 |
| chb11 | 0.600 | 0.605 | 0.608 | 0.989 | 0.828 | 0.632 | 0.781 | 0.797 |
| chb12 | 0.813 | 0.482 | 0.500 | 0.415 | 0.866 | 0.764 | 0.871 | 0.855 |
| chb13 | 0.631 | 0.470 | 0.471 | 0.629 | 0.580 | 0.726 | 0.676 | 0.681 |
| chb14 | 0.914 | 0.936 | 0.927 | 0.888 | 0.938 | 0.921 | 0.913 | 0.948 |
| chb15 | 0.721 | 0.625 | 0.637 | 0.768 | 0.840 | 0.714 | 0.841 | 0.824 |
| chb16 | 0.757 | 0.666 | 0.681 | 0.817 | 0.867 | 0.685 | 0.856 | 0.856 |
| chb17 | 0.590 | 0.370 | 0.353 | 0.994 | 0.962 | 0.659 | 0.920 | 0.947 |
| chb18 | 0.823 | 0.651 | 0.633 | 0.915 | 0.845 | 0.821 | 0.882 | 0.855 |
| chb19 | 0.990 | 0.917 | 0.914 | 0.958 | 0.953 | 0.969 | 0.970 | 0.983 |
| chb20 | 0.980 | 0.833 | 0.844 | 0.876 | 0.987 | 0.978 | 0.983 | 0.989 |
| chb22 | 0.731 | 0.609 | 0.636 | 0.998 | 0.988 | 0.828 | 0.980 | 0.983 |
| chb23 | 0.908 | 0.754 | 0.773 | 0.996 | 0.990 | 0.842 | 0.979 | 0.987 |
| chb24 | 0.866 | 0.828 | 0.833 | 0.805 | 0.913 | 0.882 | 0.907 | 0.910 |

#### Paired Wilcoxon signed-rank tests over patients (two-sided)

| Metric | A | B | n | median A − B | A better in | p |
|---|---|---|---|---|---|---|
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | – no graph | 23 | +0.144 | 21/23 | 1.19e-06 |
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | Band-power deviation | 23 | -0.056 | 3/23 | 0.000738 |
| AUROC | Band power + learned-graph change | Band-power deviation | 23 | -0.007 | 8/23 | 0.286 |
| AUROC | Band power + learned-graph change | Band power + Granger change | 23 | -0.005 | 7/23 | 0.098 |
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | Granger (per window) | 23 | +0.003 | 13/23 | 0.941 |
| AUROC | NEUROTRACE (learned graph + GAT + Transformer) | VAR forecast error only | 23 | -0.016 | 8/23 | 0.315 |
| AUROC | Prediction error only (GNN + Transformer) | VAR forecast error only | 23 | -0.144 | 2/23 | 2.1e-05 |
| AUROC | Prediction error + graph change | VAR forecast error only | 23 | -0.135 | 4/23 | 0.00112 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | – no graph | 23 | +0.067 | 20/23 | 4.94e-05 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | Band-power deviation | 23 | -0.218 | 3/23 | 6.03e-05 |
| AUPRC | Band power + learned-graph change | Band-power deviation | 23 | -0.063 | 2/23 | 2.38e-06 |
| AUPRC | Band power + learned-graph change | Band power + Granger change | 23 | -0.033 | 6/23 | 0.00112 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | Granger (per window) | 23 | +0.015 | 14/23 | 0.19 |
| AUPRC | NEUROTRACE (learned graph + GAT + Transformer) | VAR forecast error only | 23 | -0.043 | 5/23 | 0.00128 |
| AUPRC | Prediction error only (GNN + Transformer) | VAR forecast error only | 23 | -0.148 | 3/23 | 7.32e-05 |
| AUPRC | Prediction error + graph change | VAR forecast error only | 23 | -0.127 | 4/23 | 8.85e-05 |

#### Sensitivity (latency, s) at matched false-alarm rates; threshold set per test patient (oracle)

| Detector | ≤ 0.5 FA/h | ≤ 1 FA/h | ≤ 2 FA/h |
|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 0.20 ± 0.13 (17 s) | 0.26 ± 0.15 (17 s) | 0.33 ± 0.15 (18 s) |
| – diffusion GNN instead of GAT | 0.20 ± 0.13 (17 s) | 0.27 ± 0.15 (17 s) | 0.32 ± 0.15 (17 s) |
| – distance-prior graph only | 0.08 ± 0.06 (14 s) | 0.11 ± 0.08 (11 s) | 0.21 ± 0.12 (18 s) |
| – no graph | 0.09 ± 0.07 (12 s) | 0.11 ± 0.08 (11 s) | 0.21 ± 0.12 (19 s) |
| – no Transformer | 0.20 ± 0.13 (17 s) | 0.26 ± 0.15 (17 s) | 0.32 ± 0.15 (17 s) |
| – edge module without VAR evidence | 0.21 ± 0.14 (15 s) | 0.27 ± 0.15 (18 s) | 0.32 ± 0.16 (17 s) |
| – masked-window objective | 0.20 ± 0.13 (17 s) | 0.27 ± 0.15 (19 s) | 0.32 ± 0.15 (17 s) |
| – + supervised head (uses labels) | 0.69 ± 0.19 (17 s) | 0.73 ± 0.17 (16 s) | 0.74 ± 0.17 (13 s) |
| Per-channel CNN-LSTM | 0.08 ± 0.07 (13 s) | 0.10 ± 0.08 (13 s) | 0.20 ± 0.12 (17 s) |
| Band-power deviation | 0.29 ± 0.14 (18 s) | 0.42 ± 0.16 (19 s) | 0.55 ± 0.16 (16 s) |
| Pearson deviation | 0.15 ± 0.11 (14 s) | 0.23 ± 0.15 (15 s) | 0.31 ± 0.15 (16 s) |
| Coherence deviation | 0.06 ± 0.06 (18 s) | 0.10 ± 0.08 (19 s) | 0.18 ± 0.13 (15 s) |
| Granger (per window) | 0.17 ± 0.11 (13 s) | 0.20 ± 0.13 (12 s) | 0.30 ± 0.14 (20 s) |
| PDC (per window) | 0.04 ± 0.04 (14 s) | 0.13 ± 0.09 (14 s) | 0.21 ± 0.13 (15 s) |
| DTF (per window) | 0.07 ± 0.06 (12 s) | 0.16 ± 0.10 (19 s) | 0.25 ± 0.14 (17 s) |
| Band power + learned-graph change | 0.17 ± 0.12 (16 s) | 0.30 ± 0.14 (19 s) | 0.48 ± 0.15 (16 s) |
| Band power + Granger change | 0.27 ± 0.14 (18 s) | 0.39 ± 0.16 (19 s) | 0.55 ± 0.17 (18 s) |

#### Score terms: window AUROC of each term alone, and its share of the score at alarm windows

| Variant | AUROC pred. error | AUROC graph change | AUROC forecast error | share pred. | share graph | share forecast |
|---|---|---|---|---|---|---|
| NEUROTRACE (learned graph + GAT + Transformer) | 0.652 | 0.519 | 0.813 | 0.24 | 0.01 | 0.75 |
| – diffusion GNN instead of GAT | 0.641 | 0.519 | 0.814 | 0.24 | 0.01 | 0.75 |
| – distance-prior graph only | 0.648 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |
| – no graph | 0.648 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |
| – no Transformer | 0.628 | 0.519 | 0.813 | 0.23 | 0.01 | 0.76 |
| – edge module without VAR evidence | 0.653 | 0.668 | 0.816 | 0.21 | 0.01 | 0.77 |
| – masked-window objective | 0.737 | 0.511 | 0.813 | 0.24 | 0.01 | 0.75 |
| Per-channel CNN-LSTM | 0.592 | 0.500 | 0.500 | 1.00 | 0.00 | 0.00 |

#### Score-term ablation of the full model (score rebuilt from the saved terms; all three = full model)

| Score | Window AUROC | AUPRC | Sens. @ 0.5 FA/h | Sens. @ 1 FA/h | Sens. @ 2 FA/h |
|---|---|---|---|---|---|
| All three terms (= full model) | 0.800 ± 0.059 | 0.225 ± 0.090 | 0.20 | 0.26 | 0.33 |
| Prediction error only (GNN + Transformer) | 0.652 ± 0.081 | 0.096 ± 0.036 | 0.09 | 0.14 | 0.22 |
| Learned-graph change only | 0.519 ± 0.090 | 0.046 ± 0.020 | 0.02 | 0.03 | 0.07 |
| VAR forecast error only | 0.813 ± 0.060 | 0.273 ± 0.104 | 0.23 | 0.28 | 0.35 |
| Prediction error + graph change | 0.635 ± 0.079 | 0.085 ± 0.031 | 0.09 | 0.13 | 0.21 |
| Prediction error + VAR error | 0.789 ± 0.069 | 0.244 ± 0.095 | 0.21 | 0.29 | 0.35 |
| Graph change + VAR error | 0.809 ± 0.051 | 0.237 ± 0.097 | 0.21 | 0.26 | 0.30 |

What this says:
- **The simplest detector wins.** Band-power deviation reaches 0.896 ± 0.049
  window AUROC against 0.800 ± 0.059 for the full model, and is better than the full model in
  20 of 23 patients (paired Wilcoxon, significant, p < 0.001). At a matched 1 false alarm
  per hour it detects 0.42 of seizures against 0.26 for the full model.
- **The learned graph versus no graph.** With no graph or the distance prior only, the model reaches
  0.648 ± 0.077 and 0.648 ± 0.077. The full model is better than the no-graph model in
  21 of 23 patients (significant, p < 0.001); the largest gap is chb04
  (0.760 vs 0.381).
- **Where the detection comes from.** Rebuilt from one score term at a time, the VAR forecast error of
  the edge module alone reaches 0.813 AUROC (0.28 sensitivity at 1 FA/h), the
  GNN + Transformer prediction error alone 0.652 (0.14), the learned-graph change
  alone 0.519; all three together 0.800. Most of the detection therefore comes from the VAR term, not from the GNN and the Transformer:
  adding the other two terms to the VAR term changes AUROC by -0.013 (full better than VAR alone in
  8 of 23 patients, not significant, p = 0.31), and the prediction error alone is better than
  the VAR term alone in 2 of 23 patients (significant, p < 0.001). The VAR term is computed by
  the learned edge module (its directed edges and lag filters forecasting the signal inside each
  window), before the GNN and the Transformer.
- **Why the ablations alarm alike.** Each variant has its own scores and threshold. At alarm windows
  75 % of the full score is the VAR term, 24 % the prediction error and
  1 % the graph change; the GAT, Transformer, VAR-evidence and objective ablations leave the
  VAR term in place, and their alarm windows overlap 88 %–98 % (Jaccard, `alarm_overlap.csv`).
- **Band power plus graph.** Band power + learned-graph change reaches 0.901 ± 0.036 (better than band
  power alone in 8 of 23 patients, not significant, p = 0.29); band power + Granger change
  0.910 ± 0.036.
- **The supervised head** (0.893 ± 0.061) is trained on the other patients' seizure labels: an
  upper reference, not a label-free competitor.
- **Caveats.** 23 paediatric patients from one centre; intervals are wide and per-patient results vary a
  lot. Matched-FA thresholds are set on each test patient (an oracle) and only compare detectors at
  the same alarm budget.

## Limitations

- **Not a medical device.** Research and education only; no clinical validation.
- **Dependencies, not causes.** Shared inputs, filtering, the reference and volume conduction create
  directed edges without any causal pathway.
- **Electrodes, not regions.** CHB-MIT channels are bipolar derivations placed at their midpoints;
  source reconstruction is an optional extension with its own errors.
- **Window-resolution timing.** Paths are ordered at the 1 s stride; true delays are tens of
  milliseconds, so only path order is testable, and only on synthetic data.
- **Small real cohort.** 23 paediatric patients from one centre, mostly seizure files; wide intervals,
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
