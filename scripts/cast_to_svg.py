# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""An asciicast v2 recording (scripts/f2/chat.py's chat.cast) as an animated
SVG that a README shows inline, at the recording's real timing, looping.

Only what chat.cast holds is supported: printable text, CR and LF. Each
event's text appears at its time (SMIL, which browsers run in <img>), and
the whole replays after a pause.

Usage: python scripts/cast_to_svg.py CHAT.cast OUT.svg
"""

import argparse
import json
from html import escape
from pathlib import Path

FONT_PX = 14
CHAR_PX = 8.4  # a monospace advance at FONT_PX, about 0.6 em
LINE_PX = 18
PAD_PX = 12
HOLD_S = 4.0  # the finished screen, before the replay
BG, FG, USER = "#0d1117", "#e6edf3", "#7ee787"


def place(events: list[list], width: int) -> list[tuple[float, int, int, str, bool]]:
    """Each event's text cut into runs on one row: (time, row, col, text,
    from a user line), with CR, LF and wrapping at `width` columns."""
    runs, row, col, user = [], 0, 0, False
    for t, kind, text in events:
        if kind != "o":
            continue
        run, start = "", col
        for ch in text:
            if ch in "\r\n" or col == width:  # the run ends; LF and a wrap start a row
                if run:
                    runs.append((t, row, start, run, user))
                row, col = row + (ch != "\r"), 0
                run, start = "", 0
                if ch in "\r\n":
                    continue
            if col == 0:
                user = ch == ">"  # a line the user typed
            run += ch
            col += 1
        if run:
            runs.append((t, row, start, run, user))
    return runs


def svg(header: dict, events: list[list]) -> str:
    """The SVG document."""
    width = header["width"]
    runs = place(events, width)
    rows = max(r for _, r, *_ in runs) + 1
    end_s = max(t for t, *_ in runs) + HOLD_S
    w, h = 2 * PAD_PX + width * CHAR_PX, 2 * PAD_PX + rows * LINE_PX
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}" font-family="ui-monospace, SFMono-Regular, '
        f'Menlo, Consolas, monospace" font-size="{FONT_PX}">',
        f"<title>{escape(header.get('title', 'recording'))}</title>",
        f'<rect width="100%" height="100%" rx="6" fill="{BG}"/>',
        f'<rect width="0" height="0"><animate id="clock" attributeName="x" from="0" to="0" '
        f'dur="{end_s:.3f}s" begin="0s;clock.end"/></rect>',
    ]
    for t, row, col, text, user in runs:
        x, y = PAD_PX + col * CHAR_PX, PAD_PX + (row + 0.8) * LINE_PX
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{USER if user else FG}" '
            f'xml:space="preserve" visibility="hidden">{escape(text)}'
            f'<set attributeName="visibility" to="hidden" begin="clock.begin"/>'
            f'<set attributeName="visibility" to="visible" begin="clock.begin+{t:.3f}s"/>'
            f"</text>"
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cast", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    lines = args.cast.read_text().splitlines()
    args.out.write_text(svg(json.loads(lines[0]), [json.loads(line) for line in lines[1:]]))


if __name__ == "__main__":
    main()
