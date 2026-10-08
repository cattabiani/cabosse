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


# --- Shared by the checkpoint reports -------------------------------------------


def tests_passed(summary: str) -> bool:
    """Whether a pytest summary line such as '166 passed, 5 deselected in
    38.8s' has every selected test passed: none failed, errored or skipped (a
    test skips without the weights or the RTL tools, and then shows nothing)."""
    return " passed" in summary and not re.search(r"\b(failed|error|errors|skipped)\b", summary)


def met(ok: bool) -> str:
    """An exit criterion's "met" cell."""
    return "yes" if ok else "**no**"


def verdict(rows: list[list], milestone: str) -> str:
    """The verdict line from exit-criteria rows [criterion, measured, met]."""
    unmet = [row[0] for row in rows if row[2] != "yes"]
    if unmet:
        return "**Not met:** " + "; ".join(unmet) + "."
    return f"All {milestone} exit criteria are met."


def provenance_text(provenance: dict) -> str:
    """Where and when the test data was measured."""
    dirty = " with uncommitted changes" if provenance["dirty"] else ""
    return (
        f"Run on {provenance['date']}, commit `{provenance['commit'][:7]}`{dirty}, "
        f"{provenance['machine']}."
    )
