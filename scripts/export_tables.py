"""Refresh only the results tables in the visualizer's index.json (no model, no explanations).

    python scripts/export_tables.py chbmit

Copies ``results/<dataset>/summary.csv`` and the analysis CSVs of ``scripts/analyze_results.py``
into ``web/public/data/index.json``; the site's results text reads every number from there.
"""

from __future__ import annotations

import json
import sys

from export_explanations import update_index_tables

from neurotrace.experiment import to_jsonable
from neurotrace.utils import ROOT

if __name__ == "__main__":
    path = ROOT / "web" / "public" / "data" / "index.json"
    index = json.loads(path.read_text())
    for ds in sys.argv[1:] or ["chbmit"]:
        update_index_tables(index, ds)
        print(f"{ds}: {len(index['summary'][ds])} summary rows, analysis {sorted(index['analysis'][ds])}")
    path.write_text(json.dumps(to_jsonable(index), indent=1))
