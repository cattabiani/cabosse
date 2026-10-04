# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Regenerate the greedy-decode regression fixtures in model/tests/fixtures/.

Run on the reference platform, x86-64 Linux (D-023), after a deliberate
change to the numerics, and review the diff before committing. The SmolLM2
fixture needs the checkpoint and takes about 5 minutes. Examples:
    python scripts/make_fixtures.py            # both
    python scripts/make_fixtures.py --tiny     # tiny config only
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

import fixtures  # noqa: E402
import paths  # noqa: E402


def write(name: str, data: dict) -> None:
    path = paths.FIXTURES / name
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"written: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tiny", action="store_true", help="only the tiny-config fixture")
    parser.add_argument("--weights", type=Path, default=paths.SMOLLM2)
    args = parser.parse_args()
    if not paths.REFERENCE_PLATFORM:
        sys.exit(paths.REFERENCE_PLATFORM_NOTE)
    write("tiny_greedy.json", fixtures.tiny_fixture())
    if not args.tiny:
        write("smollm2_greedy.json", fixtures.smollm2_fixture(args.weights))


if __name__ == "__main__":
    main()
