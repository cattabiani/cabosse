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


def opcodes_table() -> str:
    return blocks.table(
        ["opcode", "command", "does"],
        [[int(op), f"`{op.name}`", commands.OP_DOC[op]] for op in commands.Op],
    )


def encoding_table() -> str:
    rows, offset = [], 0
    for name, fmt, meaning in commands.FIELDS:
        size = struct.calcsize("<" + fmt)
        rows.append([offset, size, f"`{name}`", meaning])
        offset += size
    return (
        blocks.table(["byte", "bytes", "field", "meaning"], rows)
        + f"\n\n{commands.COMMAND_BYTES} bytes per command, little-endian."
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
    sizes = commands.buffer_bytes(layout.config, layout.cap)
    rows = [[f"`{b.name}`", f"{n:,}"] for b, n in sizes.items()]
    total = sum(sizes.values())
    return (
        blocks.table(["buffer", "bytes"], rows)
        + f"\n\nTotal {total / 2**10:,.0f} KiB for {MODEL} with a cache of {layout.cap} positions."
    )


def fmt(cmd: commands.Command) -> list[str]:
    def buf(b: commands.Buf) -> str:
        return "" if b == commands.Buf.NONE else b.name

    return [
        cmd.op.name,
        buf(cmd.dst),
        buf(cmd.a),
        buf(cmd.b),
        str(cmd.n or ""),
        str(cmd.m or ""),
        f"0x{cmd.addr:X}" if cmd.op in commands.ADDRESSED else "",
    ]


def token_block(layout: commands.Layout) -> str:
    """The first layer in full, then the rest as a count."""
    cmds = commands.build(layout)
    c = layout.config
    first = [i for i, cmd in enumerate(cmds) if cmd.op == commands.Op.RMSNORM][::2]
    per_layer = first[1] - first[0]  # from one layer's first RMSNORM to the next
    head = cmds[: first[0] + per_layer]
    tail = cmds[first[0] + per_layer * c.num_hidden_layers :]
    header = ["#", "command", "dst", "a", "b", "n", "m", "addr"]
    rows = [[i, *fmt(cmd)] for i, cmd in enumerate(head)]
    rows.append(
        ["…", f"layers 1 to {c.num_hidden_layers - 1}: the same, at their own addresses"] + [""] * 6
    )
    rows += [[len(cmds) - len(tail) + i, *fmt(cmd)] for i, cmd in enumerate(tail)]
    size = len(cmds) * commands.COMMAND_BYTES
    return (
        f"{MODEL}, cache of {layout.cap} positions:\n\n"
        + blocks.table(header, rows)
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
