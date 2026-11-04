# Design decisions

Short records of the technical choices behind NEUROTRACE and why they were made.
Each entry: **decision**, **why**, **consequence / limit**.

## D1. Electrode-level graph first, source space later
- **Decision:** nodes are scalp electrodes (or bipolar derivations for CHB-MIT), not brain regions.
- **Why:** volume conduction spreads one cortical source over many electrodes, so calling an electrode
  a "region" overclaims. Source reconstruction (MNE, Desikan-Killiany) needs a head model and still
  leaks signal between parcels; it is an optional add-on, not the default.
- **Limit:** edges between neighbouring electrodes partly reflect shared sources, not interaction.

## D2. "Influence" means directed statistical dependency
- **Decision:** every edge is labelled as a directed statistical dependency.
- **Why:** both Granger-type baselines and attention weights measure predictability, not intervention.
  Unobserved common drivers, volume conduction and filtering can all create directed edges.

## D3. Subject-level splits only
- **Decision:** all train/validation/test splits are by subject (leave-subjects-out).
- **Why:** overlapping windows from the same recording are nearly identical; splitting by window
  leaks the test subject into training and inflates every metric.

## D4. Reproducibility
- **Decision:** one seed (`seed` in the config) seeds Python, NumPy and PyTorch; cuDNN is deterministic.
  Configs are composed with Hydra and saved next to each run's outputs.
