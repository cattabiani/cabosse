# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Fill docs/architecture.md's tables (commands, encoding, registers,
buffers, one decode step) from model/commands.py. Runs in a few seconds and
needs no weights. After changing the command set:
    python scripts/report_arch.py            # rewrite docs/architecture.md
    python scripts/report_arch.py --check    # fail if docs/architecture.md is stale
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))

from reporting import arch_doc, blocks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    blocks.write_or_check(arch_doc.PATH, arch_doc.render(), args.check)


if __name__ == "__main__":
    main()
