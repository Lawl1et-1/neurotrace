# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite + three.js, built to static files and deployed to GitHub Pages (user's choice). The Python pipeline exports JSON that the site reads; there is no live backend on the public site (FastAPI serves the same JSON locally).

## Users

Portfolio reviewers — recruiters, admissions committees and prospective supervisors — judging the author's research and engineering ability. They arrive from a CV or GitHub link, skim for a few minutes, and decide whether the work is rigorous and the person is worth talking to. Secondary: ML / neuroscience readers who check method, metrics and limitations.

## Product Purpose

NEUROTRACE is a research prototype that learns how relationships between EEG channels change over time, flags unusual changes in those relationships, and shows a dynamic, interpretable map of them. The site explains the problem, the method and the results, and lets a visitor explore real model outputs (synthetic ground-truth data and CHB-MIT seizure recordings). Success: a reviewer understands the idea in under a minute and trusts that the results are reported honestly.

## Positioning

Most EEG anomaly models score individual signals. NEUROTRACE models the *relationships* between channels over time — a learned, directed, time-varying graph (GNN) read by a temporal Transformer — and explains each flag with a propagation path, changed edges and attributions. It is checked against a synthetic generator with known couplings and against classical connectivity baselines (Granger, PDC, DTF), including where those baselines win.

## Operating Context

Visitors read on laptops and phones, usually once, often from a link. The demo runs entirely in the browser from precomputed JSON exported by `scripts/export_explanations.py`.

## Capabilities and Constraints

- Every screen must show "Research prototype — not for clinical use".
- Learned edges are "directed statistical dependency", never "causality" or "influence" without qualification.
- Electrodes / bipolar derivations are not brain regions; the site must not imply source localisation.
- Results shown are exactly those in `results/` (CSV); no rounded-up or cherry-picked claims.

## Evidence on Hand

- `results/synthetic/summary.csv`, `results/chbmit/summary.csv`, figures in `results/*/figures/`.
- Exported demo data (synthetic subject with ground truth, CHB-MIT seizure recordings).
- No users, testimonials, publications or clinical validation exist; never imply any.

## Product Principles

1. Honesty over impressiveness: show where the model loses to a baseline.
2. The map is the argument: the interactive graph should make the idea obvious faster than prose.
3. Every claim links to a number in `results/`.
4. Scientific vocabulary stays precise (dependency, not causation; electrode, not region).

## Accessibility & Inclusion

WCAG AA contrast; the demo must be keyboard-operable (timeline, event selection) and respect reduced motion.
