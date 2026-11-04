---
name: NEUROTRACE
description: EEG relationship maps, written onto one continuous chart record. Research prototype, not for clinical use.
colors:
  paper: "#f4f7f5"
  paper-deep: "#e9f0ec"
  grid-minor: "#e1ebe5"
  grid-major: "#c9dcd1"
  ink: "#14212b"
  ink-2: "#33434e"
  ink-3: "#56666f"
  blue: "#265d96"
  blue-soft: "#c9dbee"
  red: "#d23b2a"
  red-ink: "#b5301f"
  red-soft: "#f6d9d4"
  green: "#2f8f6a"
typography:
  display:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(32px, min(3.1vw, 4.7svh), 52px)"
    fontWeight: 600
    lineHeight: 0.98
    letterSpacing: "-0.005em"
  headline:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "clamp(36px, 5.2vw, 64px)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "-0.005em"
  title:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.02
  body:
    fontFamily: "Source Sans 3, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.55
  lede:
    fontFamily: "Source Sans 3, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 400
    lineHeight: 1.5
  annotation:
    fontFamily: "Source Sans 3, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 600
  label:
    fontFamily: "Barlow Condensed, Arial Narrow, sans-serif"
    fontSize: "14px"
    fontWeight: 600
    letterSpacing: "0.05em"
  code:
    fontFamily: "ui-monospace, Consolas, monospace"
    fontSize: "14px"
    lineHeight: 1.6
rounded:
  hair: "2px"
  sm: "3px"
  round: "50%"
spacing:
  minor: "6px"
  major: "30px"
  gutter: "clamp(64px, 9vw, 132px)"
  header: "52px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "11px 16px"
  button-primary-hover:
    backgroundColor: "{colors.red}"
    textColor: "#ffffff"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "11px 16px"
  button-ghost-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  toggle-segment:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
  toggle-segment-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  select:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 8px"
  legend-control-pressed:
    backgroundColor: "{colors.paper-deep}"
    textColor: "{colors.ink}"
    rounded: "{rounded.hair}"
    padding: "2px 4px"
  record-header:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    height: "{spacing.header}"
  icon-button:
    textColor: "{colors.ink}"
    rounded: "{rounded.round}"
    size: "34px"
  event-mark:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.red}"
    rounded: "{rounded.round}"
    size: "18px"
  event-mark-selected:
    backgroundColor: "{colors.red}"
    textColor: "#ffffff"
---

# Design System: NEUROTRACE

## Overview

**Creative North Star: "The Chart Record"**

The whole site is one continuous EEG chart record. Cool white chart paper is ruled in a pale green grid. Traces, type and structure are drawn in blue-black pen ink, and a red annotation pen marks only what the model found. Each section is a later page of the same recording, with a channel-label gutter on the left that carries a page code and a timestamp. Findings are not placed on cards. They are written onto the paper as a reader's annotations.

The record is dense and calm, the way an instrument printout is. Colour goes only to what moves or matters: blue for learned dependency and the anomaly score, red for detected events, edges that changed, the disclaimer stamp and annotations, and green only for the synthetic ground truth. Everything else is ink on paper. Depth comes from rules and borders on a flat sheet. Nothing floats.

The site is light only, on purpose. It is a paper record, and a dark theme would turn it into a dashboard. Motion follows the chart feed: the paper moves right to left at a steady rate, the propagation path lights up one channel at a time, and nothing else animates. With reduced motion, the feed pauses and the path appears all at once.

**Key Characteristics:**
- Ruled chart paper (6px minor / 30px major grid) as the ground for every record page.
- A left channel gutter on every page, holding a page code and a timestamp in condensed labels.
- Three inks with fixed jobs (blue dependency, red event, green ground truth) on a blue-black base ink.
- Annotations are italic semibold red text, set off by a thin red rule.
- Flat sheet: hairline and 2px ink rules, no shadows, no cards.
- Light only.

## Colors

The palette is pen and paper: a faintly green chart paper, blue-black ink, and three working inks rationed to the data.

### Primary
- **Record Ink Blue** (blue): learned directed dependency edges on the montage map, the anomaly score trace and its dashed threshold, links, and the sequential magnitude ramp (paper to this blue) in the attribution tables. Its chroma is above 0.1 on the paper ground, which keeps it readable as a data colour.

### Secondary
- **Annotation Red Pen** (red): reserved for detected event brackets and marks, edges that grew compared with the previous minute, the lit propagation path and its electrodes, the disclaimer stamp, step numbers on the path, annotation text, caveat dashes, the primary button hover and the focus ring.
- **Red Ink** (red-ink, #b5301f): the same pen for small text (annotations, path step numbers, highlighted channel names): 5.7:1 on paper, 5.3:1 on deep paper, AA at every size. `red` (#d23b2a, 4.4:1) is for marks, lines, large display text and the stamp only.
- **Red Wash** (red-soft): text selection only.

### Tertiary
- **Ground-Truth Green** (green): only for synthetic ground truth, drawn as dashed arrows on the map, a dashed interval box on the traces and a dashed seed ring. At low opacity (0.07 to 0.16) it also fills the clinicians' labelled seizure band. It is never used for status, success or decoration.

### Neutral
- **Chart Paper** (paper): page ground, ghost and select fills, idle electrodes.
- **Deep Paper** (paper-deep): pressed legend controls and the highlighted table row.
- **Minor Rule** (grid-minor) and **Major Rule** (grid-major): the 6px and 30px grid, gutter dividers, row rules and the select border.
- **Pen Ink** (ink): text, head outline, page-top rules (2px), table header rules (1.5px), primary buttons, the record header and the code block.
- **Ink 2** (ink-2): ledes, secondary prose, legend and tool labels.
- **Ink 3** (ink-3): gutter timestamps, captions, table headers, empty-state notes.
- **Soft Blue** (blue-soft): a pale tint of the record blue. It is declared but rarely used.

### Named Rules
**The Rationed Ink Rule.** Blue means a learned dependency or the score. Red means something the model detected or changed, or the disclaimer. Green means injected ground truth. No ink is ever used for anything else, including decoration, status chips or brand accents.

**The Paper Record Rule.** The site is light only. Ground is always chart paper, and the only dark surfaces are the record header and the command block, both in pen ink.

## Typography

**Display Font:** Barlow Condensed (with Arial Narrow, sans-serif)
**Body Font:** Source Sans 3 (with system-ui, sans-serif)
**Label/Mono Font:** Barlow Condensed for labels and figures; ui-monospace (Consolas) for the command block only.

**Character:** A condensed instrument-label face for anything you would find printed on a chart (headings, channel names, codes, times, numbers, controls), paired with a humanist sans for reading. Annotations use the same humanist sans in italic semibold, coloured with the red pen.

### Hierarchy
- **Display** (600, clamp(32px, min(3.1vw, 4.7svh), 52px), 0.98): the thesis in the first viewport, max 15ch. Emphasis inside it turns red without becoming italic.
- **Headline** (600, clamp(36px, 5.2vw, 64px), 1.02): one per record page, max 18ch, balanced wrapping.
- **Title** (600, 24px; 22px in the event report and pipeline): subsections and pipeline steps.
- **Body** (400, 18px / 1.55, 16px under 640px): prose, max 68ch, pretty wrapping. The lede is 20px / 1.5 in ink 2, max 60ch.
- **Label** (600, 14px, 0.05 to 0.06em, uppercase): buttons, toggles, table headers, the stamp and nav. Gutter page codes are 700, 26px.
- **Annotation** (italic 600, 16px, red-ink): margin notes beside headings and pipeline steps.

### Named Rules
**The Chart Label Rule.** Every number, channel name, time and code is set in Barlow Condensed with tabular figures. Body prose never carries data.

## Layout

Every record page is a two-column grid: the channel gutter (clamp(64px, 9vw, 132px), 52px under 640px) and the page body, with a 2px ink rule across the top. The gutter is right-aligned and holds a page code and a timestamp, divided from the body by a major-rule hairline. Body padding is clamp(28px, 5vw, 64px) on top and clamp(16px, 4vw, 56px) at the sides.

The first viewport is a three-column grid (gutter, map at 1.35fr, thesis at minmax(300px, 1fr)) above a full-width trace strip and transport bar. The traces and the timeline share the same gutter, so channel names line up with the record. The event report uses a 12-column grid with 28px by 36px gaps, split 5/7 and 7/5. The record header is sticky at 52px (48px on mobile), and its mark cell matches the gutter width.

Below 960px, the explorer becomes one column (map, transport, thesis), the explorer gutter and the header nav are hidden, and report columns go full width. Below 640px, the gutter narrows, body text drops to 16px and the header meta is hidden. The disclaimer stamp stays visible at every width.

## Elevation & Depth

The system is completely flat. There are no box shadows anywhere. Structure comes from rules of different weights: 2px ink rules start each page, 1.5px ink rules sit under table headers and on top of the pipeline, 1px major-rule hairlines divide rows and gutters, and a dashed major rule separates the map from the thesis. The only layering is the sticky ink header over the paper.

### Named Rules
**The Flat Sheet Rule.** Depth comes from ruled lines of different weights, never from shadows or lifted cards. A new surface either sits on the paper or is a dark ink band.

## Shapes

Shapes are sharp and drafted. Rectangular controls take a 3px corner, and legend and channel-name hit areas take 2px. Circles appear only where an instrument uses them: electrodes, event marks on the timeline, and round icon buttons. Lines are the main form: traces, arcs with arrowheads, brackets over events, and dashes in front of caveats. The disclaimer stamp is the one rotated element (-1.2deg), with a 1.5px red outline, like an ink stamp on the record.

## Components

### Buttons
Printed and decisive, like a stamped instruction.
- **Shape:** nearly square corners (3px), 1.5px ink border.
- **Primary:** ink fill, paper text, uppercase condensed label at 17px / 600 with 0.05em tracking, 11px by 16px padding, optional 16px inline SVG icon.
- **Hover / Focus:** the primary turns red with white text, easing over 0.2s with cubic-bezier(0.16, 1, 0.3, 1). Focus is a 2px red outline at a 3px offset.
- **Ghost:** transparent with ink text. It fills with ink on hover.

### Toggles and legend controls
- **Segmented toggle:** 1px ink outline, 3px corner, uppercase 14px labels. The pressed segment is filled with ink.
- **Legend as control:** each legend entry is a button with a 22px line swatch in the colour of the layer it switches (dashed for ground truth). When pressed or hovered it gets a deep paper fill.

### Inputs / Fields
- **Select:** paper fill, 1px major-rule border, 3px corner, 16px / 500 condensed text.
- **Timeline scrubber:** a native range input stacked invisibly over the score sparkline. When it has keyboard focus, the playhead turns red and thickens to 3px.

### Navigation
- **Record header:** a sticky 52px ink band. A gutter-width mark cell sets NEUROTRACE in 19px / 700 condensed type. Nav links are uppercase pale ink and turn paper-coloured when hovered or current. The header also carries a muted meta line and the disclaimer stamp at the far right. Nav is hidden below 960px.
- **Disclaimer stamp:** a 1.5px red outline, a 3px corner and a -1.2deg tilt, with uppercase 14px label text in a lightened red (#ff8c7c, 7.25:1 on ink) so it stays legible on the dark band. It is present on every screen.

### Record page gutter (signature)
A right-aligned left gutter holding a 26px / 700 page code and a 14px tabular timestamp, set off from the body by a hairline. It appears on every section and lines up with the trace channel names.

### Event marks and propagation path (signature)
Detected events are numbered 18px circles on the timeline (red outline, red fill when selected) and red brackets over the score row. The selected bracket runs down through every trace. Selecting an event lights the path one channel every 420ms, on the map and in the path list, where unlit rows sit at 0.35 opacity.

### Annotations and caveats
Pipeline annotations are italic semibold red text with a 1px red left rule and 10px of padding. Caveats are listed with a 16px by 2px red dash instead of a bullet, and each one leads with a 20px condensed heading.

### Tables
Full-width, with no fills. Headers are uppercase 14px ink 3 labels over a 1.5px ink rule. Rows are divided by major-rule hairlines. Numbers are right-aligned in 16px condensed type with tabular figures. The model's own row is highlighted in deep paper.

## Do's and Don'ts

### Do:
- **Do** put every new section on ruled chart paper, with the channel gutter (page code and timestamp) and a 2px ink rule on top.
- **Do** keep each ink to its job: blue for dependency and score, red for detections, changes, the stamp and annotations, dashed green for synthetic ground truth only.
- **Do** set numbers, channel names, times and controls in Barlow Condensed with tabular figures, and prose in Source Sans 3 at 68ch max.
- **Do** make legend entries the controls for the layers they describe.
- **Do** keep the disclaimer stamp in the header at every breakpoint.
- **Do** pause the chart feed and reveal the full path at once when reduced motion is requested.

### Don't:
- **Don't** add a dark theme. The record is paper, and it is light only.
- **Don't** use box shadows, lifted cards or tiled feature grids. Separate content with ruled lines.
- **Don't** use green for success or status, or red for emphasis that is not a detection, change, stamp or annotation.
- **Don't** use gradients except to rule the chart grid.
- **Don't** round rectangular controls beyond 3px. Circles are only for electrodes, event marks and icon buttons.
- **Don't** animate anything other than the chart feed and the propagation reveal.
