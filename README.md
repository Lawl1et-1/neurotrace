# NEUROTRACE

> **Research prototype — not a medical device, not for clinical use.**
> Nothing in this repository should be used to diagnose, monitor or treat anyone.

NEUROTRACE learns how the relationships between EEG channels change over time, flags unusual
changes in those relationships, and outputs a dynamic, interpretable map of them.

- **Spatial:** a graph neural network over the electrodes (10-20 system), on a graph that combines
  a distance prior with a learned, directed, time-varying adjacency.
- **Temporal:** a Transformer encoder over the sequence of per-window graph embeddings.
- **Explainability:** learned adjacency over time, attention maps, Integrated Gradients, edge masking
  and per-event "propagation paths".

Learned edges are **directed statistical dependencies, not proven causal influence**, and electrodes
are **not brain regions** (volume conduction spreads one source over many electrodes).

_Work in progress — sections below are filled in as each phase lands._
