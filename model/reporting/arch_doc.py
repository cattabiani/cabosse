# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of docs/architecture.md: the command set, its
encoding, the registers, the on-chip buffers and one decode step written out
as commands, all from model/commands.py. Filled by scripts/report_arch.py."""

import struct

import commands
import paths
import perf

from reporting import blocks

PATH = paths.REPO / "docs" / "architecture.md"
MODEL = "SmolLM2-135M-Instruct"
COLUMNS = ("dst", "a", "b", "n", "m", "addr")  # of the decode-step table


def opcodes_table() -> str:
    return blocks.table(
        ["opcode", "command", "does"],
        [[int(op), f"`{op.name}`", does] for op, (does, _) in commands.OPS.items()],
    )


def encoding_table() -> str:
    rows, offset = [], 0
    for name, (fmt, meaning) in commands.ENCODING.items():
        size = struct.calcsize("<" + fmt)
        users = [op.name for op, (_, used) in commands.OPS.items() if name in used]
        rows.append(
            [offset, size, f"`{name}`", meaning, ", ".join(users) if name != "op" else "all"]
        )
        offset += size
    return (
        blocks.table(["byte", "bytes", "field", "meaning", "used by"], rows)
        + f"\n\n{commands.COMMAND_BYTES} bytes per command, little-endian; unused fields are zero."
    )


def registers_table() -> str:
    return blocks.table(
        ["offset", "register", "access", "meaning"],
        [
            [f"`0x{off:X}`", f"`{name}`", access, meaning]
            for off, name, access, meaning in commands.REGISTERS
        ],
    )


def buffers_table(layout: commands.Layout) -> str:
    formats = commands.buffers(layout.config, layout.cap)
    rows = [
        [
            f"`{b.name}`",
            "BF16" if dtype == commands.BF16 else "FP32",
            f"{n:,}",
            f"{commands.n_bytes(n, dtype):,}",
        ]
        for b, (dtype, n) in formats.items()
    ]
    total = sum(commands.n_bytes(n, dtype) for dtype, n in formats.values())
    return (
        blocks.table(["buffer", "format", "elements", "bytes"], rows)
        + f"\n\nTotal {total / 2**10:,.0f} KiB for {MODEL} with a cache of {layout.cap} positions."
    )


def row(index: int | str, cmd: commands.Command) -> list:
    used = commands.OPS[cmd.op][1]

    def cell(name: str) -> str:
        if name not in used:
            return ""
        value = getattr(cmd, name)
        if name == "addr":
            return f"0x{value:X}"
        return value.name if isinstance(value, commands.Buf) else str(value)

    return [index, cmd.op.name, *(cell(name) for name in COLUMNS)]


def token_block(layout: commands.Layout) -> str:
    """Layer 0 in full, the other layers as a count."""
    head = commands.prologue(layout) + commands.layer_commands(layout, 0)
    tail = commands.epilogue(layout)
    cmds = commands.build(layout)
    layers, per_layer = layout.config.num_hidden_layers, len(commands.layer_commands(layout, 0))
    rows = [row(i, cmd) for i, cmd in enumerate(head)]
    rows.append(
        ["…", f"layers 1 to {layers - 1}: the same, at their own addresses"] + [""] * len(COLUMNS)
    )
    rows += [row(len(cmds) - len(tail) + i, cmd) for i, cmd in enumerate(tail)]
    size = len(cmds) * commands.COMMAND_BYTES
    return (
        f"{MODEL}, cache of {layout.cap} positions:\n\n"
        + blocks.table(["#", "command", *COLUMNS], rows)
        + f"\n\n{len(cmds)} commands ({per_layer} per layer), {size:,} bytes. "
        f"HBM in use: {layout.total_bytes / 2**20:,.1f} MiB."
    )


def generated() -> dict[str, str]:
    config = perf.load_config(MODEL)
    layout = commands.Layout.of(config, config.max_position_embeddings)
    return {
        "opcodes": opcodes_table(),
        "encoding": encoding_table(),
        "registers": registers_table(),
        "buffers": buffers_table(layout),
        "token": token_block(layout),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
