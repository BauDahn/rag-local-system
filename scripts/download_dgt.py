"""Downloads the Reglamento General de Circulación (Spain) from the BOE into the DGT corpus folder.

Usage: python scripts/download_dgt.py

The PDF is about 250 MB (it embeds the traffic-sign images), is git-ignored, and is the document
that data/corpora/dgt/eval_set.yaml was written for. Then run:
    python main.py --config configs/dgt.yaml --mode ingest
"""

import sys
import urllib.request
from pathlib import Path

URL = "https://www.boe.es/buscar/pdf/2003/BOE-A-2003-23514-consolidado.pdf"
DEST = Path(__file__).resolve().parent.parent / "data" / "corpora" / "dgt" / "raw" / "rgc.pdf"


def main() -> int:
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"Downloading {URL}\n  -> {DEST}")
    try:
        urllib.request.urlretrieve(URL, tmp)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    with open(tmp, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        tmp.unlink()
        print("The downloaded file is not a PDF; the BOE URL may have changed.", file=sys.stderr)
        return 1
    tmp.rename(DEST)
    print(f"Done ({DEST.stat().st_size / 1e6:.0f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
