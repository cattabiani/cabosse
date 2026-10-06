# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Generated blocks in Markdown files: prose written by hand, every measured
or computed number generated, so none is typed. A test per document fails if
the file and its data disagree.

A block sits between two markers in the Markdown file:
    <!-- begin: name -->
    ...generated...
    <!-- end: name -->
"""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

BLOCK = re.compile(r"(<!-- begin: (\S+) -->\n).*?(<!-- end: \2 -->)", re.DOTALL)


def fill(text: str, blocks: dict[str, str]) -> str:
    """Replace every block's content. Every block in the text must be given,
    and every given block must be in the text."""
    found = BLOCK.findall(text)
    names = [name for _, name, _ in found]
    assert sorted(names) == sorted(blocks), (names, sorted(blocks))
    return BLOCK.sub(lambda m: f"{m[1]}{blocks[m[2]]}\n{m[3]}", text)


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def write_or_check(path: Path, text: str, check: bool) -> None:
    """Write a rendered report, or with check=True exit with an error if the
    file on disk differs from it."""
    if not check:
        path.write_text(text)
        print(f"written: {path}")
    elif text != path.read_text():
        sys.exit(f"{path} is stale: render it")


def main(doc: str, path: Path, render: Callable[[], str]) -> None:
    """Command line of a report script: rewrite `path`, or with --check fail
    if it is stale. `doc` is the script's docstring."""
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    check = parser.parse_args().check  # first: --help and bad flags exit before rendering
    write_or_check(path, render(), check)
