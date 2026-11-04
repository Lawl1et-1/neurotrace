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
- **Synthetic ground truth (v2, harder):** 20 subjects, 19 damped oscillators on the 10-20 layout, sparse lagged couplings, mild volume conduction, 3 injected events each (tree of new couplings; a quarter also retune the seed rhythm). 5-fold leave-subjects-out.
  - Edge AUROC of the learned graph vs Pearson / coherence / Granger / PDC / DTF.
  - Event-edge AUROC (graph change during events vs injected edges) and propagation-path agreement (seed hit, recall, Kendall τ).
  - Window AUROC/AUPRC, sensitivity, latency, false alarms/h.
- **CHB-MIT:** 6 subjects (all seizure files + 3 seizure-free files each, ~61 h), 18 bipolar channels, leave-one-subject-out.
- **Ablations:** no graph, static distance graph, diffusion instead of GAT, no Transformer, no VAR evidence, masked objective, supervised head.
- All numbers: `results/synthetic/summary.csv`, `results/chbmit/summary.csv`.

## 5. Results (to be written from the CSVs)
- Synthetic (v2, 20 subjects, 5 folds): edge AUROC of the learned graph vs per-window estimators; event-edge AUROC; path agreement under both seed rules; window and event metrics with 95 % CIs.
- CHB-MIT (6 subjects, leave-one-subject-out): window AUROC/AUPRC, sensitivity, latency, false alarms/h with per-patient adaptive normalisation; compared with band-power and connectivity-deviation baselines.
- Headline numbers (mean ± 95 % CI over test subjects; full tables in README):
  - Synthetic (n = 20): learned edge AUROC 0.930 ± 0.021 vs Granger 0.968 ± 0.011; detection AUROC 0.909 ± 0.027 (full) vs 0.833 (no graph) vs 0.935 ± 0.021 (band-power deviation); seed hit 0.32 vs 0.48 (Granger).
  - CHB-MIT (n = 6): detection AUROC 0.840 ± 0.172 (full) vs 0.721 (no graph) vs 0.944 ± 0.074 (band-power deviation); supervised head 0.970 ± 0.021; full model 1.0 FA/h at sensitivity 0.18.
- Message: the learned graph is close to Granger on ground truth and improves the model over a no-graph ablation on both datasets, but a band-power deviation score is at least as good for detection; the contribution is the interpretable, checked graph, not a better detector.

## 6. Limitations
- Electrodes/bipolar derivations ≠ regions; edges ≠ causes.
- Window-resolution timing; only path order is testable, and only on synthetic data.
- Linear synthetic world favours Granger-type estimators.
- Six paediatric subjects from one centre; small n, wide intervals; TUH not evaluated.
- Sparsity hyper-parameter is sensitive and tuned on synthetic data.
- Not validated clinically; not a medical device.

## 7. Contribution, stated plainly
A reproducible, open pipeline that learns per-window directed dependency graphs, checks them against ground truth before real data, compares them with classical estimators on equal terms, and exposes every flag through an interactive, explained map — including the cases where simpler baselines win.
