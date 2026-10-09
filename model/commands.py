# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list of one decode step (docs/architecture.md, "Commands").

The host builds the list once per model (build()) and writes it into the
controller's command buffer; the controller runs it for every token, with the
token id and the position in registers, so the list does not change between
tokens. run() executes a list with the golden model's functions: it is the
reference for the controller, and the tests check that it gives the golden
decode step's bits, so every operation of the step has a command.

The engine has one command per matrix product (MATVEC, SCORES, VALUES); the
vector unit has primitives (D-031), chained into RMSNorm, RoPE, softmax and
SwiGLU in the spec's order of operations, so the chains give the golden
functions' bits.

Every result is computed in FP32 and rounded to BF16 when it is written to a
BF16 buffer or to the KV cache: the D-011 rounding points are where the
destination is BF16, not a step of their own.

Addresses are HBM byte addresses from a Layout. HBM is modelled as a map from
address to tensor, in the tensors' logical shape. lane_order() gives a matrix
in the engine's read order (docs/architecture.md, "Weight layout"), the
reordering the host does at load time.
"""

import enum
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
from golden import arith, decoder, dot, funcs, host, vector
from transformers import LlamaConfig

ALIGN = 4096  # HBM bytes; every tensor starts on a boundary
COMMAND_BUFFER_BYTES = 64 * 2**10  # on chip, next to the controller
BF16, F32 = torch.bfloat16, torch.float32
BEAT_BYTES = 32  # one beat of an HBM port (256 bits)
E = 4  # weights a lane takes per cycle (D-027)
LANES_PER_PORT = BEAT_BYTES // (E * BF16.itemsize)  # a beat holds E weights for each


class Op(enum.IntEnum):
    END = 0
    LOAD = 1
    MATVEC = 2
    KV_STORE = 3
    SCORES = 4
    VALUES = 5
    OUTPUT = 6
    ADD = 7
    MUL = 8
    FMA = 9
    EXP = 10
    RECIP = 11
    RSQRT = 12
    SUM = 13
    SUMSQ = 14
    MAX = 15
    ROTATE_HALF = 16


ENGINE_OPS = frozenset({Op.MATVEC, Op.SCORES, Op.VALUES})  # the rest is the vector unit's


class Flag(enum.IntFlag):
    """Modifiers of a command (FLAGS: meaning)."""

    NONE = 0
    NEG_A = 1
    NEG_B = 2
    B_PER_ROW = 4
    B_ROW = 8
    LEN_T = 16
    BY_TOKEN = 32
    BY_POSITION = 64
    SRC_BF16 = 128


FLAGS = {
    Flag.NEG_A: "use -a (an exact sign flip)",
    Flag.NEG_B: "use -b",
    Flag.B_PER_ROW: "b holds one value per row, used for the whole row",
    Flag.B_ROW: "b holds one row, used for every row",
    Flag.LEN_T: "rows are t long (the positions attention reads), not m",
    Flag.BY_TOKEN: "LOAD row `token` of the tensor at addr",
    Flag.BY_POSITION: "LOAD row `position` of the tensor at addr",
    Flag.SRC_BF16: "LOAD from BF16 (else FP32); converted to FP32 exactly",
}
# Vector-unit commands work on n rows of m elements (t with LEN_T). b is a
# buffer of the same shape, unless a B_ flag says otherwise; with b = NONE,
# b is the constant in `scalar`.
ROWS = ("n", "m", "flags")
_OPERANDS = Flag.NEG_A | Flag.NEG_B | Flag.B_PER_ROW | Flag.B_ROW | Flag.LEN_T
# The flags each command accepts; any other flag, or both flags of an
# EXCLUSIVE pair, is an error (run() stops, as for an unknown opcode).
ACCEPTS = {
    Op.LOAD: Flag.BY_TOKEN | Flag.BY_POSITION | Flag.SRC_BF16,
    Op.ADD: _OPERANDS,
    Op.MUL: _OPERANDS,
    Op.FMA: _OPERANDS,
    Op.EXP: Flag.NEG_A | Flag.LEN_T,
    Op.RECIP: Flag.NEG_A | Flag.LEN_T,
    Op.RSQRT: Flag.NEG_A | Flag.LEN_T,
    Op.SUM: Flag.LEN_T,
    Op.SUMSQ: Flag.LEN_T,
    Op.MAX: Flag.LEN_T,
    Op.ROTATE_HALF: Flag.NONE,
}
EXCLUSIVE = (Flag.B_PER_ROW | Flag.B_ROW, Flag.BY_TOKEN | Flag.BY_POSITION)

# Per command: what it does, and the fields it uses besides `op`
# (docs/architecture.md, "Commands"). `t` is position + 1, the positions
# attention reads.
OPS = {
    Op.END: ("token done: set STATUS done, raise the interrupt; the last command", ()),
    Op.LOAD: ("dst = up(tensor at addr), m elements", ("dst", "m", "flags", "addr")),
    Op.MATVEC: ("dst = W·a, W at addr: n rows × m columns", ("dst", "a", "n", "m", "addr")),
    Op.KV_STORE: (
        "cache[h][position] = a[h], n heads of m; cache at addr, cap rows per head",
        ("a", "n", "m", "cap", "addr"),
    ),
    Op.SCORES: (
        "dst[h][i] = dot(a[h], K[h // group][i]) · scalar, n heads of m, i < t",
        ("dst", "a", "n", "m", "kv_heads", "cap", "addr", "scalar"),
    ),
    Op.VALUES: (
        "dst[h] = Σᵢ a[h][i] · V[h // group][i], n heads of m, i < t",
        ("dst", "a", "n", "m", "kv_heads", "cap", "addr"),
    ),
    Op.OUTPUT: ("a (n FP32) to host memory at LOGITS_HI:LOGITS_LO", ("a", "n")),
    Op.ADD: ("dst = add(a, b)", ("dst", "a", "b", *ROWS, "scalar")),
    Op.MUL: ("dst = mul(a, b)", ("dst", "a", "b", *ROWS, "scalar")),
    Op.FMA: ("dst = fma(a, b, c)", ("dst", "a", "b", "c", *ROWS)),
    Op.EXP: ("dst = exp(a)", ("dst", "a", *ROWS)),
    Op.RECIP: ("dst = recip(a)", ("dst", "a", *ROWS)),
    Op.RSQRT: ("dst = rsqrt(a)", ("dst", "a", *ROWS)),
    Op.SUM: ("dst[r] = sum of row r of a, with S partial sums", ("dst", "a", *ROWS)),
    Op.SUMSQ: ("dst[r] = sum of squares of row r of a", ("dst", "a", *ROWS)),
    Op.MAX: ("dst[r] = max of row r of a", ("dst", "a", *ROWS)),
    Op.ROTATE_HALF: ("each row of a: (-second half, first half)", ("dst", "a", *ROWS)),
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
    R = 16  # one value per row: a sum, a max, RMSNorm's scale
    W = 17  # RMSNorm's gain
    COS = 18  # RoPE tables, this position's row
    SIN = 19
    QT = 20  # RoPE's rotated q
    E = 21  # SiLU's steps


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
        Buf.T: (F32, hidden),  # also RMSNorm's x times its scale
        Buf.G: (F32, inter),
        Buf.U: (F32, inter),
        Buf.M: (BF16, inter),
        Buf.LOGITS: (F32, c.vocab_size),
        Buf.R: (F32, c.num_attention_heads),
        Buf.W: (F32, hidden),
        Buf.COS: (F32, c.head_dim),
        Buf.SIN: (F32, c.head_dim),
        Buf.QT: (F32, q),
        Buf.E: (F32, inter),
    }


def n_bytes(shape: tuple[int, ...] | int, dtype: torch.dtype) -> int:
    return torch.Size([shape] if isinstance(shape, int) else shape).numel() * dtype.itemsize


@dataclass(frozen=True)
class Command:
    op: Op
    dst: Buf = Buf.NONE
    a: Buf = Buf.NONE
    b: Buf = Buf.NONE
    c: Buf = Buf.NONE
    flags: Flag = Flag.NONE
    kv_heads: int = 0
    n: int = 0
    m: int = 0
    cap: int = 0
    addr: int = 0
    scalar: int = 0  # FP32 bits

    def row_length(self, t: int) -> int:
        """Elements per row of a vector-unit command: t with LEN_T, else m."""
        return t if self.flags & Flag.LEN_T else self.m

    def encode(self) -> bytes:
        return struct.pack(FORMAT, *(getattr(self, f.name) for f in fields(self)))

    @staticmethod
    def decode(raw: bytes) -> Command:
        values = dict(zip(FIELD_NAMES, struct.unpack(FORMAT, raw), strict=True))
        values["op"] = Op(values["op"])
        values |= {k: Buf(values[k]) for k in ("dst", "a", "b", "c")}
        values["flags"] = Flag(values["flags"])
        return Command(**values)


# The 32-byte encoding, in field order: struct format and meaning. Little-endian.
ENCODING = {
    "op": ("B", "opcode"),
    "dst": ("B", "destination buffer"),
    "a": ("B", "first source buffer"),
    "b": ("B", "second source buffer"),
    "c": ("B", "third source buffer"),
    "flags": ("B", "modifiers (flags table)"),
    "kv_heads": ("H", "KV heads"),
    "n": ("I", "rows or heads"),
    "m": ("I", "columns, row length or elements"),
    "cap": ("I", "KV cache rows per head"),
    "addr": ("Q", "HBM byte address"),
    "scalar": ("I", "FP32 bits: a constant (scale, 1/n, eps, 1)"),
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
        shapes = {
            "embed": ((c.vocab_size, hidden), BF16),
            "rope_cos": ((cap, d), F32),
            "rope_sin": ((cap, d), F32),
        }
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
            tensors["lm_head"] = tensors["embed"]  # last: run() names the address lm_head
        return Layout(config, cap, tensors)

    def addr(self, name: str) -> int:
        return self.tensors[name][0]

    def shape(self, name: str) -> tuple[int, ...]:
        return self.tensors[name][1]

    @property
    def total_bytes(self) -> int:
        return max(a + n_bytes(s, t) for a, s, t in self.tensors.values())


def load(
    model: decoder.Model, layout: Layout, cache: decoder.KVCache | None = None
) -> dict[int, torch.Tensor]:
    """HBM as the host fills it: weights and RoPE tables (shared with the
    model, never written), and an empty KV cache, or a copy of `cache`'s
    filled positions (a prompt run by decoder.step, which writes the same
    bits as one decode step per position)."""
    names = {
        "embed": model.embed,
        "rope_cos": model.cos[: layout.cap],
        "rope_sin": model.sin[: layout.cap],
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
    if cache is not None:
        n = cache.length
        assert n <= layout.cap, (n, layout.cap)
        for i in range(layout.config.num_hidden_layers):
            for name, golden in (("k_cache", cache.k[i]), ("v_cache", cache.v[i])):
                hbm[layout.addr(f"layers.{i}.{name}")][:, :n] = golden[:, :n]
    return hbm


def lane_order(w: torch.Tensor, lanes: int) -> list[torch.Tensor]:
    """A matrix in the order the engine's ports read it (docs/architecture.md,
    "Weight layout"): rows go in passes of `lanes`; in each pass port p
    streams the beats of its LANES_PER_PORT rows, E columns of each row per
    beat, row by row within the beat. Columns past a row's end, and rows past
    the matrix's end in a port's last group, are zeros (the lanes mask or
    ignore them); a port with no rows in a pass streams nothing for it.
    Returns each port's beats, BF16 [n_beats, LANES_PER_PORT * E]."""
    assert w.dim() == 2 and w.dtype == BF16, (w.shape, w.dtype)
    assert lanes > 0 and lanes % LANES_PER_PORT == 0, lanes
    n, m = w.shape
    ports, n_groups, n_cols = lanes // LANES_PER_PORT, -(-n // LANES_PER_PORT), -(-m // E)
    padded = torch.zeros(n_groups * LANES_PER_PORT, n_cols * E, dtype=BF16)
    padded[:n, :m] = w
    # Group g (rows 4g ... 4g + 3) is port g % ports's in pass g // ports.
    beats = padded.reshape(n_groups, LANES_PER_PORT, n_cols, E).transpose(1, 2)
    beats = beats.reshape(n_groups, n_cols, LANES_PER_PORT * E)
    return [beats[p::ports].reshape(-1, LANES_PER_PORT * E) for p in range(ports)]


def load_vector(dst: Buf, layout: Layout, name: str, flags: Flag = Flag.NONE) -> Command:
    _, shape, dtype = layout.tensors[name]
    by_row = flags & (Flag.BY_TOKEN | Flag.BY_POSITION)
    m = shape[-1] if by_row else math.prod(shape)
    src = Flag.SRC_BF16 if dtype == BF16 else Flag.NONE
    return Command(Op.LOAD, dst, m=m, flags=flags | src, addr=layout.addr(name))


def prologue(layout: Layout) -> list[Command]:
    """The token's embedding, and this position's RoPE tables for every layer."""
    return [
        load_vector(Buf.H, layout, "embed", Flag.BY_TOKEN),
        load_vector(Buf.COS, layout, "rope_cos", Flag.BY_POSITION),
        load_vector(Buf.SIN, layout, "rope_sin", Flag.BY_POSITION),
    ]


def norm(layout: Layout, name: str) -> list[Command]:
    """X = rmsnorm(H): mul(g, mul(x, rsqrt(add(mul(sumsq(x), 1/n), eps))))."""
    n = layout.config.hidden_size
    inv_n, eps = (f32_bits(x) for x in vector.rmsnorm_constants(n, layout.config.rms_norm_eps))
    row, one = dict(n=1, m=n), dict(n=1, m=1)
    return [
        load_vector(Buf.W, layout, name),
        Command(Op.SUMSQ, Buf.R, Buf.H, **row),
        Command(Op.MUL, Buf.R, Buf.R, scalar=inv_n, **one),
        Command(Op.ADD, Buf.R, Buf.R, scalar=eps, **one),
        Command(Op.RSQRT, Buf.R, Buf.R, **one),
        Command(Op.MUL, Buf.T, Buf.H, Buf.R, flags=Flag.B_PER_ROW, **row),
        Command(Op.MUL, Buf.X, Buf.T, Buf.W, **row),
    ]


def rope(dst: Buf, a: Buf, rotated: Buf, heads: int, d: int) -> list[Command]:
    """dst = fma(a, cos, mul(rotate_half(a), sin)), per head."""
    shape = dict(n=heads, m=d)
    return [
        Command(Op.ROTATE_HALF, rotated, a, **shape),
        Command(Op.MUL, rotated, rotated, Buf.SIN, flags=Flag.B_ROW, **shape),
        Command(Op.FMA, dst, a, Buf.COS, c=rotated, flags=Flag.B_ROW, **shape),
    ]


def softmax(heads: int) -> list[Command]:
    """P = softmax(S) per head: mul(e, recip(sum(e))), e = exp(add(s, -max(s)))."""
    rows, per_row = dict(n=heads), Flag.LEN_T | Flag.B_PER_ROW  # rows t long
    return [
        Command(Op.MAX, Buf.R, Buf.S, flags=Flag.LEN_T, **rows),
        Command(Op.ADD, Buf.S, Buf.S, Buf.R, flags=per_row | Flag.NEG_B, **rows),
        Command(Op.EXP, Buf.S, Buf.S, flags=Flag.LEN_T, **rows),
        Command(Op.SUM, Buf.R, Buf.S, flags=Flag.LEN_T, **rows),
        Command(Op.RECIP, Buf.R, Buf.R, m=1, **rows),
        Command(Op.MUL, Buf.P, Buf.S, Buf.R, flags=per_row, **rows),
    ]


def swiglu(n: int) -> list[Command]:
    """M = mul(silu(G), U), silu(a) = mul(a, recip(add(exp(-a), 1)))."""
    row = dict(n=1, m=n)
    return [
        Command(Op.EXP, Buf.E, Buf.G, flags=Flag.NEG_A, **row),
        Command(Op.ADD, Buf.E, Buf.E, scalar=f32_bits(1.0), **row),
        Command(Op.RECIP, Buf.E, Buf.E, **row),
        Command(Op.MUL, Buf.E, Buf.G, Buf.E, **row),
        Command(Op.MUL, Buf.M, Buf.E, Buf.U, **row),
    ]


def matvec(layout: Layout, dst: Buf, name: str, a: Buf) -> Command:
    rows, cols = layout.shape(name)
    return Command(Op.MATVEC, dst, a, n=rows, m=cols, addr=layout.addr(name))


def layer_commands(layout: Layout, i: int) -> list[Command]:
    """One layer, in the order of golden.decoder.step()."""
    c = layout.config
    d, heads, kv_heads = c.head_dim, c.num_attention_heads, c.num_key_value_heads
    p = f"layers.{i}."
    attention = dict(n=heads, m=d, kv_heads=kv_heads, cap=layout.cap)

    def kv_store(name: str, a: Buf) -> Command:
        return Command(Op.KV_STORE, a=a, n=kv_heads, m=d, cap=layout.cap, addr=layout.addr(name))

    scale = f32_bits(host.attention_scale(d))
    residual = Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=1, m=c.hidden_size)
    return [
        *norm(layout, p + "attn_norm"),
        matvec(layout, Buf.Q, p + "q", Buf.X),
        matvec(layout, Buf.K, p + "k", Buf.X),
        matvec(layout, Buf.V, p + "v", Buf.X),
        *rope(Buf.KR, Buf.K, Buf.KR, kv_heads, d),
        kv_store(p + "k_cache", Buf.KR),
        kv_store(p + "v_cache", Buf.V),
        *rope(Buf.QR, Buf.Q, Buf.QT, heads, d),
        Command(
            Op.SCORES, Buf.S, Buf.QR, addr=layout.addr(p + "k_cache"), scalar=scale, **attention
        ),
        *softmax(heads),
        Command(Op.VALUES, Buf.ATT, Buf.P, addr=layout.addr(p + "v_cache"), **attention),
        matvec(layout, Buf.T, p + "o", Buf.ATT),
        residual,
        *norm(layout, p + "mlp_norm"),
        matvec(layout, Buf.G, p + "gate", Buf.X),
        matvec(layout, Buf.U, p + "up", Buf.X),
        *swiglu(c.intermediate_size),
        matvec(layout, Buf.T, p + "down", Buf.M),
        residual,
    ]


def epilogue(layout: Layout) -> list[Command]:
    vocab = layout.config.vocab_size
    return [
        *norm(layout, "final_norm"),
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


def flags_ok(cmd: Command) -> bool:
    """Only flags the command accepts, and at most one of each EXCLUSIVE pair."""
    accepted = ACCEPTS.get(cmd.op, Flag.NONE)
    return not cmd.flags & ~accepted and all((cmd.flags & pair) != pair for pair in EXCLUSIVE)


@dataclass(frozen=True)
class EngineCall:
    """One engine command as run() executed it: what the engine reads and
    what it writes, bit for bit (the RTL tests' real data)."""

    index: int  # in the command list
    name: str  # the Layout name of what it reads in HBM
    cmd: Command
    a: torch.Tensor  # the input buffer, BF16
    mem: torch.Tensor  # the matrix, or a copy of the cache's first t positions per head
    out: torch.Tensor  # dst as written: FP32, or rounded where the buffer is BF16


def run(
    commands: list[Command],
    layout: Layout,
    hbm: dict[int, torch.Tensor],
    token: int,
    position: int,
    record: Callable[[EngineCall], None] | None = None,
) -> torch.Tensor | None:
    """Execute a command list for one token, as the controller does: buffers
    start empty, HBM (the KV cache) is updated in place. Returns what OUTPUT
    wrote to the host, or None for a list without OUTPUT (a prompt token
    whose logits the host does not need). `record` gets every engine
    command's EngineCall."""
    assert len(commands) * COMMAND_BYTES <= COMMAND_BUFFER_BYTES, len(commands)
    formats = buffers(layout.config, layout.cap)
    names = {addr: name for name, (addr, _, _) in layout.tensors.items()} if record else {}
    buf: dict[Buf, torch.Tensor] = {}
    output = None
    t = position + 1  # positions attention reads: 0 ... position

    def write(dst: Buf, x: torch.Tensor) -> None:
        """Store an FP32 result, rounded if the buffer is BF16."""
        dtype, length = formats[dst]
        assert x.dtype == F32 and x.numel() <= length, (dst, x.dtype, x.numel(), length)
        buf[dst] = arith.bf16(x) if dtype == BF16 else x

    def rows(cmd: Command, x: Buf) -> torch.Tensor:
        """Buffer x as the command's n rows."""
        return buf[x].reshape(cmd.n, cmd.row_length(t))

    def operand_a(cmd: Command) -> torch.Tensor:
        v = rows(cmd, cmd.a)
        return -v if cmd.flags & Flag.NEG_A else v  # unary minus: an exact sign flip

    def operand_b(cmd: Command) -> torch.Tensor:
        if cmd.b == Buf.NONE:
            v = f32_value(cmd.scalar)
        elif cmd.flags & Flag.B_PER_ROW:
            v = buf[cmd.b].reshape(cmd.n, 1)
        elif cmd.flags & Flag.B_ROW:
            v = buf[cmd.b].reshape(1, -1)
        else:
            v = rows(cmd, cmd.b)
        return -v if cmd.flags & Flag.NEG_B else v

    unary = {Op.EXP: funcs.exp, Op.RECIP: funcs.recip, Op.RSQRT: funcs.rsqrt}
    binary = {Op.ADD: arith.add, Op.MUL: arith.mul}
    reduce = {
        Op.SUM: vector.reduce_sum,
        Op.SUMSQ: vector.reduce_sum_squares,
        Op.MAX: vector.reduce_max,
    }
    for i, cmd in enumerate(commands):
        assert cmd.op in OPS, f"unknown command {cmd.op!r}"  # stops, as case _ below
        assert flags_ok(cmd), f"command {i}: flags {cmd.flags!r} not accepted by {cmd.op!r}"
        a, mem = buf.get(cmd.a), hbm.get(cmd.addr)
        match cmd.op:
            case Op.LOAD:
                assert (mem.dtype == BF16) == bool(cmd.flags & Flag.SRC_BF16), (mem.dtype, cmd)
                if cmd.flags & Flag.BY_TOKEN:
                    mem = mem[token]
                elif cmd.flags & Flag.BY_POSITION:
                    mem = mem[position]
                assert mem.numel() == cmd.m, (mem.shape, cmd.m)
                write(cmd.dst, arith.up(mem.reshape(-1)) if mem.dtype == BF16 else mem.reshape(-1))
            case Op.MATVEC:
                assert mem.shape == (cmd.n, cmd.m) and a.dtype == BF16, (mem.shape, a.dtype)
                write(cmd.dst, dot.matvec(mem, a[None])[0])  # one vector
            case Op.KV_STORE:
                assert mem.shape == (cmd.n, cmd.cap, cmd.m) and a.dtype == F32, (mem.shape, a.dtype)
                mem[:, position] = arith.bf16(a.reshape(cmd.n, cmd.m))  # the cache is BF16
            case Op.SCORES:
                q = a.reshape(
                    cmd.kv_heads, cmd.n // cmd.kv_heads, cmd.m
                )  # head h: KV head h // group
                s = decoder.attention_scores(q, mem[:, None, :t], f32_value(cmd.scalar))
                write(cmd.dst, s.reshape(-1))
            case Op.VALUES:
                p = a.reshape(cmd.kv_heads, cmd.n // cmd.kv_heads, t)
                write(cmd.dst, decoder.attention_values(p, mem[:, None, :t]).reshape(-1))
            case Op.OUTPUT:
                assert a.shape == (cmd.n,) and a.dtype == F32, (a.shape, a.dtype)
                output = a.clone()
            case Op.ADD | Op.MUL:
                write(cmd.dst, binary[cmd.op](operand_a(cmd), operand_b(cmd)).reshape(-1))
            case Op.FMA:
                fma = arith.fma(operand_a(cmd), operand_b(cmd), rows(cmd, cmd.c))
                write(cmd.dst, fma.reshape(-1))
            case Op.EXP | Op.RECIP | Op.RSQRT:
                write(cmd.dst, unary[cmd.op](operand_a(cmd)).reshape(-1))
            case Op.SUM | Op.SUMSQ | Op.MAX:
                write(cmd.dst, reduce[cmd.op](rows(cmd, cmd.a)))
            case Op.ROTATE_HALF:
                write(cmd.dst, vector.rotate_half(rows(cmd, cmd.a)).reshape(-1))
            case Op.END:  # must be the last command (the COMMANDS register's count)
                assert i == len(commands) - 1, f"{len(commands) - 1 - i} commands after END"
                return output
            case _:  # the controller stops with STATUS error and the index in ERROR
                raise AssertionError(f"unknown command {cmd.op!r}")
        if record is not None and cmd.op in ENGINE_OPS:
            read = mem if cmd.op == Op.MATVEC else mem[:, :t].clone()  # later tokens write it
            record(EngineCall(i, names[cmd.addr], cmd, a.clone(), read, buf[cmd.dst].clone()))
    raise AssertionError("command list without END")


def record_step(
    model: decoder.Model, prompt: list[int], token: int
) -> tuple[torch.Tensor, list[EngineCall]]:
    """One decode step of `token` at position len(prompt), after `prompt`
    (run by decoder.step, its KV cache copied into HBM): the logits and the
    step's engine commands as run() executed them."""
    layout = Layout.of(model.config, cap=len(prompt) + 1)
    cache = decoder.KVCache.empty(model, layout.cap)
    if prompt:
        decoder.step(model, cache, prompt)
    hbm, calls = load(model, layout, cache), []
    logits = run(build(layout), layout, hbm, token, len(prompt), calls.append)
    assert logits is not None
    return logits, calls
