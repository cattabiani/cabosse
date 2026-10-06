# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list of one decode step (docs/architecture.md, "Commands").

The host builds the list once per model (build()) and writes it into the
controller's command buffer; the controller runs it for every token, with the
token id and the position in registers, so the list does not change between
tokens. run() executes a list with the golden model's functions: it is the
reference for the controller, and the tests check that it gives the golden
decode step's bits, so every operation of the step has a command.

Every result is computed in FP32 and rounded to BF16 when it is written to a
BF16 buffer or to the KV cache: the D-011 rounding points are where the
destination is BF16, not a step of their own.

Addresses are HBM byte addresses from a Layout. HBM is modelled as a map from
address to tensor, in the tensors' logical shape: the engine's read order
(docs/architecture.md, "Weight layout") is a reordering the model leaves out.
"""

import enum
import struct
from dataclasses import dataclass, fields

import torch
from golden import arith, decoder, dot, host, vector
from transformers import LlamaConfig

ALIGN = 4096  # HBM bytes; every tensor starts on a boundary
COMMAND_BUFFER_BYTES = 64 * 2**10  # on chip, next to the controller
BF16, F32 = torch.bfloat16, torch.float32


class Op(enum.IntEnum):
    END = 0
    EMBED = 1
    RMSNORM = 2
    MATVEC = 3
    ROPE = 4
    KV_STORE = 5
    SCORES = 6
    SOFTMAX = 7
    VALUES = 8
    SWIGLU = 9
    ADD = 10
    OUTPUT = 11


# Per command: what it does, and the fields it uses besides `op`
# (docs/architecture.md, "Commands"). `t` is position + 1, the positions
# attention reads.
OPS = {
    Op.END: ("token done: set STATUS done, raise the interrupt; the last command", ()),
    Op.EMBED: ("dst = up(table[token]), rows of m; table at addr", ("dst", "m", "addr")),
    Op.RMSNORM: (
        "dst = rmsnorm(a, gain at addr, eps = scalar), n elements",
        ("dst", "a", "n", "addr", "scalar"),
    ),
    Op.MATVEC: ("dst = W·a, W at addr: n rows × m columns", ("dst", "a", "n", "m", "addr")),
    Op.ROPE: (
        "dst = rope(a), n vectors of m; cos and sin at addr, row position",
        ("dst", "a", "n", "m", "addr"),
    ),
    Op.KV_STORE: (
        "cache[h][position] = a[h], n heads of m; cache at addr, cap rows per head",
        ("a", "n", "m", "cap", "addr"),
    ),
    Op.SCORES: (
        "dst[h][i] = dot(a[h], K[h // group][i]) · scalar, n heads of m, i < t",
        ("dst", "a", "n", "m", "kv_heads", "cap", "addr", "scalar"),
    ),
    Op.SOFTMAX: ("dst = softmax(a), n vectors of t", ("dst", "a", "n")),
    Op.VALUES: (
        "dst[h] = Σᵢ a[h][i] · V[h // group][i], n heads of m, i < t",
        ("dst", "a", "n", "m", "kv_heads", "cap", "addr"),
    ),
    Op.SWIGLU: ("dst = swiglu(a, b), n elements", ("dst", "a", "b", "n")),
    Op.ADD: ("dst = add(a, b), n elements", ("dst", "a", "b", "n")),
    Op.OUTPUT: ("a (n FP32) to host memory at LOGITS_HI:LOGITS_LO", ("a", "n")),
}


class Buf(enum.IntEnum):
    """On-chip buffers, named by what they hold (BUFFERS: format and size)."""

    NONE = 0
    H = 1  # residual stream
    X = 2  # normalized input of a matrix product
    Q = 3
    K = 4
    V = 5
    QR = 6  # q after RoPE
    KR = 7  # k after RoPE
    S = 8  # attention scores
    P = 9  # attention probabilities
    ATT = 10  # attention output
    T = 11  # a product to add to the residual stream
    G = 12  # gate
    U = 13  # up
    M = 14  # SwiGLU output
    LOGITS = 15


def buffers(config: LlamaConfig, cap: int) -> dict[Buf, tuple[torch.dtype, int]]:
    """Each buffer's format and length in elements, for a model and a KV
    cache of `cap` positions."""
    c = config
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    hidden, inter, scores = c.hidden_size, c.intermediate_size, c.num_attention_heads * cap
    return {
        Buf.H: (F32, hidden),
        Buf.X: (BF16, hidden),
        Buf.Q: (F32, q),
        Buf.K: (F32, kv),
        Buf.V: (F32, kv),
        Buf.QR: (BF16, q),
        Buf.KR: (F32, kv),  # rounded when stored in the cache
        Buf.S: (F32, scores),
        Buf.P: (BF16, scores),
        Buf.ATT: (BF16, q),
        Buf.T: (F32, hidden),
        Buf.G: (F32, inter),
        Buf.U: (F32, inter),
        Buf.M: (BF16, inter),
        Buf.LOGITS: (F32, c.vocab_size),
    }


def n_bytes(shape: tuple[int, ...] | int, dtype: torch.dtype) -> int:
    return torch.Size([shape] if isinstance(shape, int) else shape).numel() * dtype.itemsize


@dataclass(frozen=True)
class Command:
    op: Op
    dst: Buf = Buf.NONE
    a: Buf = Buf.NONE
    b: Buf = Buf.NONE
    n: int = 0
    m: int = 0
    kv_heads: int = 0
    cap: int = 0
    addr: int = 0
    scalar: int = 0  # FP32 bits

    def encode(self) -> bytes:
        return struct.pack(FORMAT, *(getattr(self, f.name) for f in fields(self)))

    @staticmethod
    def decode(raw: bytes) -> Command:
        values = dict(zip(FIELD_NAMES, struct.unpack(FORMAT, raw), strict=True))
        values["op"] = Op(values["op"])
        values |= {k: Buf(values[k]) for k in ("dst", "a", "b")}
        return Command(**values)


# The 32-byte encoding, in field order: struct format and meaning. Little-endian.
ENCODING = {
    "op": ("B", "opcode"),
    "dst": ("B", "destination buffer"),
    "a": ("B", "first source buffer"),
    "b": ("B", "second source buffer"),
    "n": ("I", "rows, elements, vectors or heads"),
    "m": ("I", "columns or vector length"),
    "kv_heads": ("I", "KV heads"),
    "cap": ("I", "KV cache rows per head"),
    "addr": ("Q", "HBM byte address"),
    "scalar": ("I", "FP32 bits: eps or scale"),
}
FIELD_NAMES = tuple(f.name for f in fields(Command))
assert tuple(ENCODING) == FIELD_NAMES, (tuple(ENCODING), FIELD_NAMES)
FORMAT = "<" + "".join(fmt for fmt, _ in ENCODING.values())
COMMAND_BYTES = struct.calcsize(FORMAT)

# Host-visible registers on the shell's OCL port: offset, name, access, meaning.
REGISTERS = [
    (0x00, "ID", "R", "0x43424F53 ('CBOS')"),
    (0x04, "VERSION", "R", "architecture version"),
    (0x08, "CONTROL", "W", "bit 0: start a token; bit 1: reset the controller"),
    (0x0C, "STATUS", "R", "bit 0: busy; bit 1: done; bit 2: error"),
    (0x10, "TOKEN", "RW", "token id of the next step"),
    (0x14, "POSITION", "RW", "position of the next step"),
    (0x18, "COMMANDS", "RW", "number of commands in the list"),
    (0x1C, "ERROR", "R", "index of the command that failed, and why"),
    (0x20, "LOGITS_LO", "RW", "host address for OUTPUT, low 32 bits"),
    (0x24, "LOGITS_HI", "RW", "host address for OUTPUT, high 32 bits"),
    (0x28, "CYCLES_LO", "R", "cycles of the last token, low 32 bits"),
    (0x2C, "CYCLES_HI", "R", "cycles of the last token, high 32 bits"),
    (0x10000, "COMMAND_BUFFER", "W", "the command list: 64 KiB, up to 2,048 commands"),
]


def f32_bits(x: float | torch.Tensor) -> int:
    """The FP32 bits of x (a float is rounded to FP32 once)."""
    return int(arith.bits_f32(torch.as_tensor(x, dtype=F32).reshape(1))[0])


def f32_value(bits: int) -> torch.Tensor:
    return arith.f32_from_bits(torch.tensor([bits], dtype=torch.int64))[0]


@dataclass(frozen=True)
class Layout:
    """Where each tensor sits in HBM: name -> (address, shape, dtype). The KV
    cache has `cap` rows per KV head and layer."""

    config: LlamaConfig
    cap: int
    tensors: dict[str, tuple[int, tuple[int, ...], torch.dtype]]

    @staticmethod
    def of(config: LlamaConfig, cap: int) -> Layout:
        c = config
        assert 1 <= cap <= c.max_position_embeddings, (cap, c.max_position_embeddings)
        d, hidden, inter = c.head_dim, c.hidden_size, c.intermediate_size
        q, kv = c.num_attention_heads * d, c.num_key_value_heads * d
        cache = (c.num_key_value_heads, cap, d)
        shapes = {"embed": ((c.vocab_size, hidden), BF16), "rope": ((cap, 2, d), F32)}
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}."
            shapes |= {
                p + "attn_norm": ((hidden,), BF16),
                p + "q": ((q, hidden), BF16),
                p + "k": ((kv, hidden), BF16),
                p + "v": ((kv, hidden), BF16),
                p + "o": ((hidden, q), BF16),
                p + "mlp_norm": ((hidden,), BF16),
                p + "gate": ((inter, hidden), BF16),
                p + "up": ((inter, hidden), BF16),
                p + "down": ((hidden, inter), BF16),
                p + "k_cache": (cache, BF16),
                p + "v_cache": (cache, BF16),
            }
        shapes["final_norm"] = ((hidden,), BF16)
        if not c.tie_word_embeddings:
            shapes["lm_head"] = ((c.vocab_size, hidden), BF16)
        tensors, addr = {}, 0
        for name, (shape, dtype) in shapes.items():
            tensors[name] = (addr, shape, dtype)
            addr += -(-n_bytes(shape, dtype) // ALIGN) * ALIGN
        if c.tie_word_embeddings:
            tensors["lm_head"] = tensors["embed"]
        return Layout(config, cap, tensors)

    def addr(self, name: str) -> int:
        return self.tensors[name][0]

    def shape(self, name: str) -> tuple[int, ...]:
        return self.tensors[name][1]

    @property
    def total_bytes(self) -> int:
        return max(a + n_bytes(s, t) for a, s, t in self.tensors.values())


def load(model: decoder.Model, layout: Layout) -> dict[int, torch.Tensor]:
    """HBM as the host fills it: weights and RoPE tables (shared with the
    model, never written), and an empty KV cache."""
    names = {
        "embed": model.embed,
        "rope": torch.stack([model.cos, model.sin], dim=1)[: layout.cap],
        "final_norm": model.final_norm,
        "lm_head": model.lm_head,
    }
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}."
        names |= {p + f.name: getattr(layer, f.name) for f in fields(layer)}
    hbm = {}
    for name, (addr, shape, dtype) in layout.tensors.items():
        t = names[name] if name in names else torch.zeros(shape, dtype=dtype)
        assert t.shape == shape and t.dtype == dtype, (name, t.shape, shape, t.dtype)
        hbm[addr] = t
    return hbm


def prologue(layout: Layout) -> list[Command]:
    c = layout.config
    return [Command(Op.EMBED, Buf.H, m=c.hidden_size, addr=layout.addr("embed"))]


def norm(layout: Layout, name: str) -> Command:
    c = layout.config
    eps = f32_bits(c.rms_norm_eps)
    return Command(Op.RMSNORM, Buf.X, Buf.H, n=c.hidden_size, addr=layout.addr(name), scalar=eps)


def matvec(layout: Layout, dst: Buf, name: str, a: Buf) -> Command:
    rows, cols = layout.shape(name)
    return Command(Op.MATVEC, dst, a, n=rows, m=cols, addr=layout.addr(name))


def layer_commands(layout: Layout, i: int) -> list[Command]:
    """One layer, in the order of golden.decoder.step()."""
    c = layout.config
    d, heads, kv_heads = c.head_dim, c.num_attention_heads, c.num_key_value_heads
    p = f"layers.{i}."
    rope = layout.addr("rope")
    attention = dict(n=heads, m=d, kv_heads=kv_heads, cap=layout.cap)

    def kv_store(name: str, a: Buf) -> Command:
        return Command(Op.KV_STORE, a=a, n=kv_heads, m=d, cap=layout.cap, addr=layout.addr(name))

    scale = f32_bits(host.attention_scale(d))
    return [
        norm(layout, p + "attn_norm"),
        matvec(layout, Buf.Q, p + "q", Buf.X),
        matvec(layout, Buf.K, p + "k", Buf.X),
        matvec(layout, Buf.V, p + "v", Buf.X),
        Command(Op.ROPE, Buf.KR, Buf.K, n=kv_heads, m=d, addr=rope),
        kv_store(p + "k_cache", Buf.KR),
        kv_store(p + "v_cache", Buf.V),
        Command(Op.ROPE, Buf.QR, Buf.Q, n=heads, m=d, addr=rope),
        Command(
            Op.SCORES, Buf.S, Buf.QR, addr=layout.addr(p + "k_cache"), scalar=scale, **attention
        ),
        Command(Op.SOFTMAX, Buf.P, Buf.S, n=heads),
        Command(Op.VALUES, Buf.ATT, Buf.P, addr=layout.addr(p + "v_cache"), **attention),
        matvec(layout, Buf.T, p + "o", Buf.ATT),
        Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=c.hidden_size),
        norm(layout, p + "mlp_norm"),
        matvec(layout, Buf.G, p + "gate", Buf.X),
        matvec(layout, Buf.U, p + "up", Buf.X),
        Command(Op.SWIGLU, Buf.M, Buf.G, Buf.U, n=c.intermediate_size),
        matvec(layout, Buf.T, p + "down", Buf.M),
        Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=c.hidden_size),
    ]


def epilogue(layout: Layout) -> list[Command]:
    vocab = layout.config.vocab_size
    return [
        norm(layout, "final_norm"),
        matvec(layout, Buf.LOGITS, "lm_head", Buf.X),
        Command(Op.OUTPUT, a=Buf.LOGITS, n=vocab),
        Command(Op.END),
    ]


def build(layout: Layout) -> list[Command]:
    """One decode step, in the order of golden.decoder.step()."""
    layers = range(layout.config.num_hidden_layers)
    return (
        prologue(layout) + [c for i in layers for c in layer_commands(layout, i)] + epilogue(layout)
    )


def run(
    commands: list[Command], layout: Layout, hbm: dict[int, torch.Tensor], token: int, position: int
) -> torch.Tensor | None:
    """Execute a command list for one token, as the controller does: buffers
    start empty, HBM (the KV cache) is updated in place. Returns what OUTPUT
    wrote to the host, or None for a list without OUTPUT (a prompt token
    whose logits the host does not need)."""
    assert len(commands) * COMMAND_BYTES <= COMMAND_BUFFER_BYTES, len(commands)
    formats = buffers(layout.config, layout.cap)
    buf: dict[Buf, torch.Tensor] = {}
    output = None
    t = position + 1  # positions attention reads: 0 ... position

    def write(dst: Buf, x: torch.Tensor) -> None:
        """Store an FP32 result, rounded if the buffer is BF16."""
        dtype, length = formats[dst]
        assert x.dtype == F32 and x.numel() <= length, (dst, x.dtype, x.numel(), length)
        buf[dst] = arith.bf16(x) if dtype == BF16 else x

    for i, cmd in enumerate(commands):
        a, b, mem = buf.get(cmd.a), buf.get(cmd.b), hbm.get(cmd.addr)
        match cmd.op:
            case Op.EMBED:
                assert mem.shape[1] == cmd.m, (mem.shape, cmd.m)
                write(cmd.dst, arith.up(mem[token]))
            case Op.RMSNORM:
                write(cmd.dst, vector.rmsnorm(a, mem, float(f32_value(cmd.scalar))))
            case Op.MATVEC:
                assert mem.shape == (cmd.n, cmd.m) and a.dtype == BF16, (mem.shape, a.dtype)
                write(cmd.dst, dot.matvec(mem, a[None])[0])  # one vector
            case Op.ROPE:
                cos, sin = mem[position]
                write(cmd.dst, vector.rope(a.reshape(cmd.n, cmd.m), cos, sin).reshape(-1))
            case Op.KV_STORE:
                assert mem.shape == (cmd.n, cmd.cap, cmd.m) and a.dtype == F32, (mem.shape, a.dtype)
                mem[:, position] = arith.bf16(a.reshape(cmd.n, cmd.m))  # the cache is BF16
            case Op.SCORES:
                q = a.reshape(
                    cmd.kv_heads, cmd.n // cmd.kv_heads, cmd.m
                )  # head h: KV head h // group
                s = decoder.attention_scores(q, mem[:, None, :t], f32_value(cmd.scalar))
                write(cmd.dst, s.reshape(-1))
            case Op.SOFTMAX:
                write(cmd.dst, vector.softmax(a.reshape(cmd.n, t)).reshape(-1))
            case Op.VALUES:
                p = a.reshape(cmd.kv_heads, cmd.n // cmd.kv_heads, t)
                write(cmd.dst, decoder.attention_values(p, mem[:, None, :t]).reshape(-1))
            case Op.SWIGLU:
                write(cmd.dst, vector.swiglu(a, b))
            case Op.ADD:
                write(cmd.dst, arith.add(a, b))
            case Op.OUTPUT:
                assert a.shape == (cmd.n,) and a.dtype == F32, (a.shape, a.dtype)
                output = a.clone()
            case Op.END:  # must be the last command (the COMMANDS register's count)
                assert i == len(commands) - 1, f"{len(commands) - 1 - i} commands after END"
                return output
            case _:  # the controller stops with STATUS error and the index in ERROR
                raise AssertionError(f"unknown command {cmd.op!r}")
    raise AssertionError("command list without END")
