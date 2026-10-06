# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The generated tables of docs/architecture.md: the command set, its
encoding, the registers, the on-chip buffers and one decode step written out
as commands, all from model/commands.py. Filled by scripts/report_arch.py."""

import struct

import commands
import paths
import perf
import torch

from reporting import blocks

PATH = paths.REPO / "docs" / "architecture.md"
MODEL = "SmolLM2-135M-Instruct"  # the decode step written out
# The model ladder after the tiny configs (D-014), each at its full context:
# the on-chip buffers are sized for the largest of each.
LADDER = ("SmolLM2-135M-Instruct", "Qwen2.5-0.5B-Instruct")
COLUMNS = ("dst", "a", "b", "c", "n", "m", "flags", "scalar", "addr")  # of the decode-step table


def opcodes_table() -> str:
    def accepts(op: commands.Op) -> str:
        return " ".join(f.name for f in commands.ACCEPTS.get(op, commands.Flag.NONE))

    return blocks.table(
        ["opcode", "command", "does", "flags"],
        [[int(op), f"`{op.name}`", does, accepts(op)] for op, (does, _) in commands.OPS.items()],
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


def flags_table() -> str:
    return blocks.table(
        ["bit", "flag", "meaning"],
        [
            [flag.bit_length() - 1, f"`{flag.name}`", meaning]
            for flag, meaning in commands.FLAGS.items()
        ],
    )


def registers_table() -> str:
    return blocks.table(
        ["offset", "register", "access", "meaning"],
        [
            [f"`0x{off:X}`", f"`{name}`", access, meaning]
            for off, name, access, meaning in commands.REGISTERS
        ],
    )


def ladder_layouts() -> dict[str, commands.Layout]:
    configs = {name: perf.load_config(name) for name in LADDER}
    return {name: commands.Layout.of(c, c.max_position_embeddings) for name, c in configs.items()}


def capacities() -> dict[commands.Buf, tuple[torch.dtype, int, str]]:
    """Each buffer at the largest size the ladder needs, and which model sets it."""
    out = {}
    for name, layout in ladder_layouts().items():
        for b, (dtype, n) in commands.buffers(layout.config, layout.cap).items():
            if b not in out or n > out[b][1]:
                out[b] = (dtype, n, name)
    return out


def buffers_table() -> str:
    caps = capacities()
    rows = [
        [
            f"`{b.name}`",
            "BF16" if dtype == commands.BF16 else "FP32",
            f"{n:,}",
            f"{commands.n_bytes(n, dtype):,}",
            name,
        ]
        for b, (dtype, n, name) in caps.items()
    ]
    total = sum(commands.n_bytes(n, dtype) for dtype, n, _ in caps.values())
    return (
        blocks.table(["buffer", "format", "elements", "bytes", "sized by"], rows)
        + f"\n\nTotal {total / 2**20:,.2f} MiB."
    )


def ladder_table() -> str:
    rows = []
    for name, layout in ladder_layouts().items():
        n = len(commands.build(layout))
        rows.append(
            [
                name,
                layout.config.num_hidden_layers,
                layout.cap,
                n,
                f"{n * commands.COMMAND_BYTES:,}",
                f"{layout.total_bytes / 2**20:,.1f}",
            ]
        )
    return blocks.table(
        ["model", "layers", "context", "commands", "command bytes", "HBM (MiB)"], rows
    ) + (
        f"\n\nCommand buffer: {commands.COMMAND_BUFFER_BYTES // 2**10} KiB, "
        f"{commands.COMMAND_BUFFER_BYTES // commands.COMMAND_BYTES:,} commands."
    )


def row(index: int | str, cmd: commands.Command) -> list:
    used = commands.OPS[cmd.op][1]

    def cell(name: str) -> str:
        if name not in used:
            return ""
        value = getattr(cmd, name)
        if name == "addr":
            return f"0x{value:X}"
        if name == "flags":
            return " ".join(f.name for f in value)
        if name == "scalar":  # an FP32 constant; 0 when unused
            return f"{float(commands.f32_value(value)):.6g}" if value else ""
        if isinstance(value, commands.Buf):
            return "" if value == commands.Buf.NONE else value.name
        return str(value)

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
        "flags": flags_table(),
        "registers": registers_table(),
        "buffers": buffers_table(),
        "ladder": ladder_table(),
        "token": token_block(layout),
    }


def render() -> str:
    return blocks.fill(PATH.read_text(), generated())
