---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: []
---

# NEUROTRACE site — surface brief

Scope: the public site (landing + interactive demo) in `web/`, one page, deployed to GitHub Pages.
Mode: Persuade (first viewport, method, results), with an Operate-grade demo inside it.
Audience: portfolio reviewers skimming from a CV link; secondary ML/neuro readers.
Action: explore the live map (play an event, click it) and open the code on GitHub.
Proof: real exported model outputs (synthetic with ground truth, CHB-MIT), numbers from `results/`.
Constraints: "Research prototype — not for clinical use" always visible; dependency, not causation; electrodes, not regions. Avoid: generic AI startup page; dry paper-PDF look.

## Direction contract

THESIS: The page is one continuous EEG chart record and NEUROTRACE's findings are written onto it as a reader's annotations. It refuses the split hero + feature tiles + metrics row of an ML project page.

OWN-WORLD: Cool white chart paper (#f4f7f5) ruled in pale green major/minor grid; traces and type in blue-black pen ink (#14212b), secondary ink blue (#2b5d8a); red annotation pen (#d23b2a) reserved for detected events, changed edges, and the disclaimer stamp. Channel-label gutter on the left of every record page; time ticks along the top rule; annotations are handwritten-feel callouts drawn with leader lines, set in a humanist grotesk, figures tabular. Colour rationed to what moves (from Rietveld).

STORY: A reviewer sees a real recording playing, the directed graph changing on the montage diagram, and a red mark where the model flags an event; they click it, watch the propagation path light channel by channel (from Versailles), then scroll through timestamped record pages — problem, method, results with honest baseline comparisons, limitations — and leave knowing what was built and how well it works.

FIRST VIEWPORT: Record header strip across the top: project name, recording ID, montage, a red stamp "Research prototype — not for clinical use". Left 58%: montage head diagram (2D topomap; 3D toggle) with directed edges live. Right 42%: the thesis written as the first annotation, two actions (Explore the event, View code). Bottom 40%: 18 EEG traces scrolling on the ruled grid with the channel gutter, the anomaly score as the top trace, red event bracket; legend items are controls (from character sheet).

FORM: The Chart Record, position 6 of my ordered list (1 seismic bulletin, 2 transit diagram, 3 synoptic chart, 4 orchestral score, 5 Swiss science poster, 6 EEG chart record, 7 Cajal ink plates). seed:0d01920b056a0480. Raises: sections are successive timestamped pages of one recording (coil tower); sequenced propagation reveal (Versailles); legend-as-control (character sheet); colour only on what moves (Rietveld); every event and time addressable by URL hash (HyperCard). Signature interaction: click an event mark → path lights in order on the map and traces, with arrival times annotated. Motion grammar: the chart feed — paper advances right-to-left at a steady rate; nothing else animates except the propagation reveal; reduced-motion pauses the feed.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
