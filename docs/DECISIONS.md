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
- **Decision:** the Transformer predicts the band power of the next window (causal attention; `objective=masked` reconstructs masked windows instead) and is trained only on sequences with no labelled window. Anomaly score = robust z(next-window error) + λ_graph · robust z(|A_t − A_t−1|) + 0.5 · robust z(within-window forecast error), with a trailing 5-window moving average so it stays usable online. Robust statistics and the alarm threshold come from the validation subject's normal windows, never from the test subject.
- **Why:** band power is predictable enough to make "surprise" meaningful, while raw EEG two seconds ahead is mostly unpredictable noise.

## D9. Baselines are first-class
- **Decision:** every table reports per-window Pearson, coherence, Granger, PDC and DTF (edge recovery, plus a connectivity-change anomaly score), a band-power deviation score and a per-channel CNN-LSTM forecaster.
- **Consequence:** on both datasets a simple band-power deviation score is a strong anomaly detector, and connectivity-change scores are near or below chance (the graph is *stable* during an event; it changes at onset and offset). Both are reported as found.

## D10. Explanations at window resolution
- **Decision:** propagation paths are ordered at the 1 s stride, from the learned graph (greedy walk over edges that grew vs the preceding minute) and, separately, from band power (first time each channel leaves its baseline).
- **Why/limit:** injected delays are 15–40 ms, far below the stride, so only the order of a path can be checked against ground truth, and on real EEG not even that.

## D11. Visualizer as a static site
- **Decision:** React + Vite + three.js reading pre-exported JSON (`scripts/export_explanations.py`), deployed to GitHub Pages; FastAPI serves the same files locally.
- **Why:** reviewers open a link and see real model output with no server to keep alive. The JSON is quantised (adjacency 0–99, traces as z × 25 at 64 Hz) and split per recording, so the first load is small.
