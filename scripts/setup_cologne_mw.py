#!/usr/bin/env python3
"""
Build indexes/mw_index.json from Cologne Sanskrit Lexicon Monier-Williams XML.

Upstream: https://www.sanskrit-lexicon.uni-koeln.de/
Download: https://www.sanskrit-lexicon.uni-koeln.de/downloads/ (mw.xml or similar)

Usage:
  python scripts/setup_cologne_mw.py --xml path/to/mw.xml
  python scripts/setup_cologne_mw.py --xml downloads/mw.xml --out indexes/mw_index.json

This replaces live Cologne API calls (blocked on many hosts). Used by mw_grounding.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Build local MW index from Cologne XML")
    p.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to Monier-Williams XML from sanskrit-lexicon.uni-koeln.de",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "indexes" / "mw_index.json",
    )
    args = p.parse_args()

    if not args.xml.exists():
        print(
            "XML not found. Download from:\n"
            "  https://www.sanskrit-lexicon.uni-koeln.de/downloads/\n"
            "See docs/LEXICAL_RESOURCES.md"
        )
        sys.exit(1)

    build_script = ROOT / "build_mw_index.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--xml",
        str(args.xml),
        "--out",
        str(args.out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    print("Done:", args.out)


if __name__ == "__main__":
    main()
