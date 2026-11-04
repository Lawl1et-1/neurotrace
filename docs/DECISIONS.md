# Design decisions

Short records of the technical choices behind NEUROTRACE and why they were made.
Each entry: **decision**, **why**, **consequence / limit**.

## D1. Electrode-level graph first, source space later
- **Decision:** nodes are scalp electrodes (synthetic, EEG-MMI) or bipolar derivations (CHB-MIT, TUH), not brain regions. A bipolar channel is drawn at the midpoint of its two electrodes.
- **Why:** volume conduction spreads one cortical source over many electrodes, so calling an electrode a "region" overclaims. Source reconstruction (MNE, Desikan-Killiany) needs a head model and still leaks signal between parcels; it is an optional extension, not the default.
- **Limit:** edges between neighbouring electrodes partly reflect shared sources, not interaction.

## D2. "Influence" means directed statistical dependency
- **Decision:** every edge is labelled as a directed statistical dependency (`adj[src, dst]`: the past of `src` helps predict `dst`).
- **Why:** both Granger-type baselines and the learned gates measure predictability, not intervention. Unobserved common drivers, volume conduction, filtering and the reference can all create directed edges.

## D3. Subject-level splits only
- **Decision:** all train/validation/test splits are by subject (leave-subjects-out). A test enforces it.
- **Why:** overlapping windows from the same recording are nearly identical; splitting by window leaks the test subject into training and inflates every metric.

## D4. Reproducibility
- **Decision:** one seed (`seed` in the config) seeds Python, NumPy and PyTorch; cuDNN is deterministic. Configs are composed with Hydra; every checkpoint stores its resolved config and its split.

## D5. Preprocessing
- **Decision:** 0.5–50 Hz zero-phase FIR band-pass, notch at the mains frequency (60 Hz for CHB-MIT) and its harmonics, resample to 128 Hz, 2 s windows with a 1 s stride. Each channel is z-scored with **robust, recording-level** statistics (median / IQR).
- **Why:** per-window normalisation would erase the amplitude changes that are part of what an anomaly looks like; robust statistics keep a few minutes of seizure from dominating an hour-long file.

## D6. How the learned graph is learned
- **Decision:** a per-window edge network scores every ordered pair from (a) node embeddings, (b) lagged cross-correlations in both directions and (c) *conditional* evidence from a ridge VAR fitted to the window in closed form (Wald statistic of each lag block plus the coefficients). The gates are trained only by a within-window forecasting loss (each channel's next sample from its own past and the gated, lagged past of the others) plus an L1 penalty. The GNN reads the gates (`detach_graph: true`) but does not train them.
- **Why:** the first version used only (a) and (b) and reached an edge AUROC of 0.55–0.81 on synthetic data while per-window Granger reached 0.97–0.99: pairwise correlations cannot separate a direct coupling from a shared input. Adding conditional evidence and a stronger sparsity penalty brought it to ~0.94. Detaching keeps the meaning of an edge fixed ("helps forecast"); in practice it made no measurable difference.
- **Consequence:** the learned graph is close to, not better than, per-window Granger on linear synthetic data. Its value is that it is amortised (one forward pass for every window), smoothed by training across subjects, and feeds the GNN directly.

## D7. Sparsity is a sensitive hyper-parameter
- **Decision:** `lambda_sparse = 0.03`.
- **Why:** swept on synthetic data: 0.001 → edge AUROC ~0.58, 0.02 → ~0.91, 0.03 → ~0.94, 0.05 and 0.1 → all gates collapse to zero (AUROC ~0.45). Chosen on synthetic data and transferred unchanged to CHB-MIT, where there is no ground truth to tune it.

## D8. Self-supervised objective and anomaly score
- **Decision:** the Transformer predicts the band power of the next window (causal attention; `objective=masked` reconstructs masked windows instead) and is trained only on sequences with no labelled window. Anomaly score = robust z(next-window error) + λ_graph · robust z(graph deviation from the preceding minute, D13) + 0.5 · robust z(within-window forecast error), with a trailing 5-window moving average so it stays usable online. Robust statistics and the alarm threshold come from the validation subject's normal windows, never from the test subject.
- **Why:** band power is predictable enough to make "surprise" meaningful, while raw EEG two seconds ahead is mostly unpredictable noise.

## D9. Baselines are first-class
- **Decision:** every table reports per-window Pearson, coherence, Granger, PDC and DTF (edge recovery, plus a connectivity-deviation anomaly score), a band-power deviation score and a per-channel CNN-LSTM forecaster, scored with exactly the same thresholds, matching and statistics as the model.

## D10. Explanations at window resolution
- **Decision:** propagation paths are ordered at the 1 s stride, from the learned graph (greedy walk over edges that grew vs the preceding minute) and, separately, from band power (first time each channel leaves its baseline).
- **Why/limit:** injected delays are 15–40 ms, far below the stride, so only the order of a path can be checked against ground truth, and on real EEG not even that.

## D11. Visualizer as a static site
- **Decision:** React + Vite + three.js reading pre-exported JSON (`scripts/export_explanations.py`), deployed to GitHub Pages; FastAPI serves the same files locally.
- **Why:** reviewers open a link and see real model output with no server to keep alive. The JSON is quantised (adjacency 0–99, traces as z × 25 at 64 Hz) and split per recording, so the first load is small.

## D12. Per-patient adaptive normalisation of the score (CHB-MIT)
- **Decision:** on CHB-MIT each score component is z-scored against the same recording's preceding 600 windows (10 min, current window excluded) before the components are combined; the alarm threshold is still fixed on the validation patient's normal windows.
- **Why:** the first fold showed a validation patient (chb03) whose normal scores were ~5× those of the test patient (chb05): the threshold transferred from chb03 never fired on chb05 (no alarm at all on the test patient). Trailing normalisation is causal (usable online) and uses no labels.
- **Limit:** a long seizure inside the trailing window lowers later z-scores; the first 30 windows of a file have no baseline and score 0.

## D13. Evaluation audit: connectivity score and event matching
- **Findings:** (1) the connectivity anomaly score |A_t − A_t−1| sat at the same level everywhere (per-window estimation noise dominates consecutive differences) and *dropped* during events, when a strong coupling makes estimates more stable: window AUROCs of 0.2–0.4 were a design error, not a sign error. (2) Event matching accepted alarms up to 15 s before onset as detections, giving negative latencies.
- **Decision:** the connectivity score — for the baselines and for the model's own graph term — is now the deviation of the (3-window smoothed) graph from the mean graph of the preceding minute (`baseline_deviation`). An event counts as detected only if the score crosses the threshold from the window straddling the onset onwards (latency ≥ −1 window); earlier crossings are false alarms. Unit tests cover a known easy case (a coupling switching on: Granger deviation AUROC > 0.9, latency 0–3 s), label alignment and the matching rules.
- **Reporting:** summaries give mean, sd, n and a 95 % t-interval per metric over test subjects (subjects are the unit; events and hours are pooled within a subject first).

## D14. Seed rule for propagation paths
- **Finding:** on v1 synthetic data the learned-graph path found the true seed in 25 % of events vs 89 % for Granger. Diagnosis: the Granger Wald statistic grows with coupling strength (change on the seed's out-edges ≈ 228 vs ≈ 99 on other event edges vs ≈ 0.3 on non-edges), so "largest net outgoing increase" finds the seed; the learned gates are bounded sigmoids (0.015 vs 0.012 vs −0.003), so every new coupling is just "on" and the rule picks nodes with many children. Seeding from the earliest window where the score rises did not help (19 %): the score rises in the first event window (median lag 0.5 windows) and all event edges switch on within one 1 s window, while true delays are 15–40 ms.
- **Decision:** paths start from `hybrid` = band-power deviation × (normalised outgoing increase + 0.2) (53 % on v1 data, chosen on those test folds, so its first out-of-sample check is the v2 rerun); `net` is still reported for every estimator.

## D15. Harder synthetic benchmark (v2)
- **Decision:** 20 subjects (5 folds), weaker seed drive (×1.2), weaker event couplings (0.25–0.4), more volume conduction (0.25) and only a quarter of events retune the seed to a slow rhythm.
- **Why:** in v1 a band-power deviation score reached 0.98 window AUROC: events were detectable from spectra alone and could not separate methods. On a 3-subject probe, v2-like settings bring band power to ≈ 0.82–0.92 and Granger deviation to ≈ 0.80–0.83, while Granger edge recovery stays ≈ 0.95.
