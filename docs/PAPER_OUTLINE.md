# NEUROTRACE — paper outline

*Working title:* **Learning time-varying directed dependency graphs from scalp EEG: a ground-truth check and an honest look at seizure detection.**

> Research prototype. Nothing here is a clinical claim.

## 1. Problem
- EEG anomaly / seizure detectors mostly score each channel (or a flattened montage) independently; they say *that* something changed, not *how the relationships between channels* changed.
- Goal: a model that (i) estimates a directed, time-varying dependency graph between channels, (ii) flags unusual changes in it, and (iii) explains each flag (which edges changed, in what order the change appeared).
- Two traps the design must avoid: treating electrodes as brain regions (volume conduction) and reading predictive dependency as causation.

## 2. Related work
- **Spatio-temporal GNNs for EEG:** graph convolution over electrode graphs for seizure detection/classification (e.g. DCRNN-style diffusion convolution with distance or correlation graphs, Tang et al. 2022; GNN surveys for EEG), graph attention (GAT) variants, learned or dynamic adjacency in traffic/time-series GNNs (Graph WaveNet, MTGNN).
- **Classical connectivity:** Granger causality, Partial Directed Coherence (Baccalá & Sameshima 2001), Directed Transfer Function (Kamiński & Blinowska 1991), coherence; their known sensitivity to volume conduction and common inputs.
- **Amortised causal / dependency discovery:** neural relational inference and amortised causal discovery (Kipf et al. 2018; Löwe et al. 2022) — inferring a graph per sample with a shared network.
- **Temporal Transformers for physiological signals** and self-supervised anomaly detection by forecasting / masked reconstruction.
- **Explainability:** Integrated Gradients (Sundararajan et al. 2017), GNNExplainer and edge masking (Ying et al. 2019), attention as (weak) explanation.

## 3. Method
1. Preprocessing (MNE): band-pass, notch, resample, 2 s / 1 s windows, robust recording-level z-score, log band power.
2. Directed edge module per window: node embeddings + lagged cross-correlations + closed-form ridge-VAR evidence → gate `A_t[src, dst]` and lag filters; trained by within-window forecasting with an L1 penalty.
3. Spatial GNN (GATConv or directed diffusion convolution) on `mix · A_prior + (1 − mix) · A_t`, distance prior from 10-20 coordinates.
4. Causal Transformer over 8 graph embeddings; next-window band-power prediction (or masked reconstruction); optional supervised head.
5. Anomaly score = prediction error + graph change + within-window forecast error (robust z, causal smoothing); threshold from held-out normal data.
6. Explanations: graph-change propagation path, band-power onset path, edge masking, Integrated Gradients (channel × window × band), attention.

## 4. Experiments
- **Synthetic ground truth:** 12 subjects, 19 damped oscillators on the 10-20 layout, sparse lagged couplings, mild volume conduction, 3 injected events each (seed rhythm + tree of new couplings). 4-fold leave-subjects-out.
  - Edge AUROC of the learned graph vs Pearson / coherence / Granger / PDC / DTF.
  - Event-edge AUROC (graph change during events vs injected edges) and propagation-path agreement (seed hit, recall, Kendall τ).
  - Window AUROC/AUPRC, sensitivity, latency, false alarms/h.
- **CHB-MIT:** 6 subjects (all seizure files + 3 seizure-free files each, ~61 h), 18 bipolar channels, leave-one-subject-out.
- **Ablations:** no graph, static distance graph, diffusion instead of GAT, no Transformer, no VAR evidence, masked objective, supervised head.
- All numbers: `results/synthetic/summary.csv`, `results/chbmit/summary.csv`.

## 5. Results (to be written from the CSVs)
- Synthetic: learned edges ≈0.94 edge AUROC, just below per-window conditional Granger (≈0.97, the right estimator for a linear VAR world) and above PDC/DTF/coherence/Pearson; removing the VAR evidence drops it to ≈0.73.
- Synthetic anomaly detection: all self-supervised variants ≈0.95 window AUROC; the graph adds no measurable detection benefit there; a band-power deviation score reaches ≈0.98.
- Propagation paths: graph-based paths find the seed in roughly a quarter of events; Granger-based paths in most of them — explanation quality lags edge recovery.
- CHB-MIT: see `results/chbmit/summary.csv`; compare with the band-power baseline (≈0.97 window AUROC) and connectivity-change baselines (≈chance).

## 6. Limitations
- Electrodes/bipolar derivations ≠ regions; edges ≠ causes.
- Window-resolution timing; only path order is testable, and only on synthetic data.
- Linear synthetic world favours Granger-type estimators.
- Six paediatric subjects from one centre; small n, wide intervals; TUH not evaluated.
- Sparsity hyper-parameter is sensitive and tuned on synthetic data.
- Not validated clinically; not a medical device.

## 7. Contribution, stated plainly
A reproducible, open pipeline that learns per-window directed dependency graphs, checks them against ground truth before real data, compares them with classical estimators on equal terms, and exposes every flag through an interactive, explained map — including the cases where simpler baselines win.
