"""Download a subset of CHB-MIT from PhysioNet (open access, no login).

By default: every file that contains a seizure plus 3 seizure-free files per subject.
Use ``--all`` to download every file (about 43 GB for the 23 patients).

    python scripts/download_chbmit.py --subjects chb01 chb02 --background 3
"""

from __future__ import annotations

import argparse

from neurotrace.data.datasets.chbmit import download_subject
from neurotrace.utils import DATA_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=[f"chb{i:02d}" for i in range(1, 25)],
                   help="default: all 24 recordings (23 patients; chb21 is chb01 again)")
    p.add_argument("--background", type=int, default=3, help="seizure-free files per subject")
    p.add_argument("--all", action="store_true", help="download every file of each subject")
    p.add_argument("--raw-dir", default=str(DATA_DIR / "raw" / "chbmit"))
    a = p.parse_args()
    for s in a.subjects:
        print(f"[{s}]")
        download_subject(s, a.raw_dir, None if a.all else a.background)


if __name__ == "__main__":
    main()
