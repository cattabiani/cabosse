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

Addresses come from a Layout (D-040): what the engine reads (matrices, KV
caches) is spread over every channel at the same offset, which is the
command's address; what LOAD reads sits in channel 0 above it, at its HBM
byte address. HBM is modelled as a map from address to tensor, in the
tensors' logical shape. lane_order() gives a matrix in the engine's read
order (docs/architecture.md, "Weight layout"), the reordering the host does
at load time; channel_writes() gives the bytes the host writes.
"""

import enum
import math
import struct
from collections.abc import Callable, Iterator
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
VALUES_HEADS = 2  # heads per p.V pass, at most (rtl/engine.sv's Heads)
STRIDE_ALIGN = 16  # elements: S and P's stride, the vector unit's width (D-030, D-049)
# Field widths of the RTL's defaults, for validate() (rtl/engine.sv, rtl/vec_unit.sv).
ROW_BITS = 20  # rows, n, m, cap, stride, token, positions
COL_BITS = 16  # the engine's columns
HEAD_BITS = 8  # attention's heads and KV heads
HEAD_DIMS = (16, 32, 64, 128)  # the engine's: a power of two, 16 to 128

VECTOR_LANES = 16  # the vector unit's elements a cycle (D-030), half a LOAD beat in FP32
ID = 0x43424F53  # 'CBOS', the ID register
VERSION = 1  # of the registers and the command set
CONTROL_START, CONTROL_RESET = 1, 2
STATUS_BUSY, STATUS_DONE, STATUS_ERROR = 1, 2, 4
ERROR_OPCODE, ERROR_MEMORY, ERROR_NO_END = 1, 2, 3  # ERROR's reason, bits 31:28
CACHE_ALIGN = 128  # bytes: a KV cache's address (rtl/vec_unit.sv)
PORTS = 32  # F2's HBM ports, one channel each (docs/f2.md)
CHANNEL_BYTES = 2**29  # F2's channel: 16 GiB over 32


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
    V_STORE = 17


ENGINE_OPS = frozenset({Op.MATVEC, Op.SCORES, Op.VALUES})  # the rest is the vector unit's
STORE_OPS = frozenset({Op.KV_STORE, Op.V_STORE})  # write a KV cache: K's, V's (D-062)


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
    Flag.LEN_T: "rows are t long (the positions attention reads), at stride STRIDE, not m",
    Flag.BY_TOKEN: "LOAD row `token` of the tensor at addr",
    Flag.BY_POSITION: "LOAD row `position` of the tensor at addr",
    Flag.SRC_BF16: "LOAD from BF16 (else FP32); converted to FP32 exactly",
}
# Vector-unit commands work on n rows of m elements, packed; with LEN_T, on
# n rows of t at the STRIDE register's stride (head h's scores at
# S[h·stride], where SCORES writes them). b is a buffer of the same shape,
# unless a B_ flag says otherwise; with b = NONE, b is the constant in
# `scalar`.
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
        "K cache[h][position] = a[h], n heads of m; cache at addr, cap rows per head",
        ("a", "n", "m", "cap", "addr"),
    ),
    Op.SCORES: (
        "dst[h][i] = dot(a[h], K[h // group][i]) · scalar, n heads of m, i < t (rows at STRIDE)",
        ("dst", "a", "n", "m", "kv_heads", "cap", "addr", "scalar"),
    ),
    Op.VALUES: (
        "dst[h] = Σᵢ a[h][i] · V[h // group][i], n heads of m, i < t (rows of a at STRIDE)",
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
    Op.V_STORE: (
        "as KV_STORE, into a V cache (stored as Vᵀ, D-062)",
        ("a", "n", "m", "cap", "addr"),
    ),
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


def buffers(config: LlamaConfig, stride: int) -> dict[Buf, tuple[torch.dtype, int]]:
    """Each buffer's format and length in elements, for a model and S and P
    rows at `stride`."""
    c = config
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    hidden, inter, scores = c.hidden_size, c.intermediate_size, c.num_attention_heads * stride
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
    (
        0x08,
        "CONTROL",
        "W",
        "bit 0: start a token; bit 1: reset the controller and the units (wins over bit 0)",
    ),
    (0x0C, "STATUS", "R", "bit 0: busy; bit 1: done; bit 2: error"),
    (0x10, "TOKEN", "RW", "token id of the next step"),
    (0x14, "POSITION", "RW", "position of the next step"),
    (0x18, "COMMANDS", "RW", "number of commands in the list"),
    (
        0x1C,
        "ERROR",
        "R",
        "31:28 why (1 opcode, 2 memory, 3 no END), 15:0 the command (memory: where the run was)",
    ),
    (0x20, "LOGITS_LO", "RW", "host address for OUTPUT, low 32 bits"),
    (0x24, "LOGITS_HI", "RW", "host address for OUTPUT, high 32 bits"),
    (0x28, "CYCLES_LO", "R", "cycles of the last token, low 32 bits"),
    (0x2C, "CYCLES_HI", "R", "cycles of the last token, high 32 bits"),
    (0x30, "STRIDE", "RW", "row stride of S and P, in elements (SCORES, VALUES, LEN_T)"),
    (0x10000, "COMMAND_BUFFER", "W", "the command list: 64 KiB, up to 2,048 commands"),
]
REG = {name: offset for offset, name, _, _ in REGISTERS}


def error_value(reason: int, index: int) -> int:
    """The ERROR register: why the controller stopped, at which command."""
    return reason << 28 | index


def f32_bits(x: float | torch.Tensor) -> int:
    """The FP32 bits of x (a float is rounded to FP32 once)."""
    return int(arith.bits_f32(torch.as_tensor(x, dtype=F32).reshape(1))[0])


def f32_value(bits: int) -> torch.Tensor:
    return arith.f32_from_bits(torch.tensor([bits], dtype=torch.int64))[0]


class Kind(enum.Enum):
    """How a tensor sits in HBM (D-040): LOAD reads it from channel 0 as it
    is, or the engine reads it spread over every channel at one offset, a
    matrix or a KV cache in the lanes' order (lane_order, kv_lane_order)."""

    LOADED = enum.auto()
    MATRIX = enum.auto()
    K_CACHE = enum.auto()
    V_CACHE = enum.auto()


CACHES = frozenset({Kind.K_CACHE, Kind.V_CACHE})
WEIGHTS = frozenset(Kind) - CACHES


@dataclass(frozen=True)
class Layout:
    """Where each tensor sits in HBM: name -> (address, shape, dtype), and
    its Kind, on an engine of `ports` ports and channels of `channel_bytes`
    (D-040). What the engine reads sits at the same offset in every channel,
    in [0, engine_bytes), reserving its largest share there; the tensors LOAD
    reads sit in channel 0 above them, up to end_bytes, so a channel-0
    offset is also the HBM address. The KV cache has `cap` rows per KV head
    and layer. Also the STRIDE register's value: S and P hold head h's row
    of t at h·stride, on chip (by default `cap` rounded up to STRIDE_ALIGN;
    any multiple of it from the longest t up works)."""

    config: LlamaConfig
    cap: int
    stride: int
    tensors: dict[str, tuple[int, tuple[int, ...], torch.dtype]]
    kinds: dict[str, Kind]
    ports: int
    channel_bytes: int
    engine_bytes: int
    end_bytes: int

    @staticmethod
    def of(
        config: LlamaConfig,
        cap: int,
        stride: int | None = None,
        ports: int = PORTS,
        channel_bytes: int = CHANNEL_BYTES,
    ) -> Layout:
        """Raises ValueError when channel 0 cannot hold it all."""
        c = config
        assert 1 <= cap <= c.max_position_embeddings, (cap, c.max_position_embeddings)
        if stride is None:
            stride = -(-cap // STRIDE_ALIGN) * STRIDE_ALIGN
        assert stride >= 1 and stride % STRIDE_ALIGN == 0, stride
        d, hidden, inter = c.head_dim, c.hidden_size, c.intermediate_size
        q, kv = c.num_attention_heads * d, c.num_key_value_heads * d
        cache = (c.num_key_value_heads, cap, d)
        ld, mat = Kind.LOADED, Kind.MATRIX
        shapes = {
            "embed": ((c.vocab_size, hidden), BF16, ld),
            "rope_cos": ((cap, d), F32, ld),
            "rope_sin": ((cap, d), F32, ld),
        }
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}."
            shapes |= {
                p + "attn_norm": ((hidden,), BF16, ld),
                p + "q": ((q, hidden), BF16, mat),
                p + "k": ((kv, hidden), BF16, mat),
                p + "v": ((kv, hidden), BF16, mat),
                p + "o": ((hidden, q), BF16, mat),
                p + "mlp_norm": ((hidden,), BF16, ld),
                p + "gate": ((inter, hidden), BF16, mat),
                p + "up": ((inter, hidden), BF16, mat),
                p + "down": ((hidden, inter), BF16, mat),
                p + "k_cache": (cache, BF16, Kind.K_CACHE),
                p + "v_cache": (cache, BF16, Kind.V_CACHE),
            }
        # Its own copy when tied to embed: LOAD reads embed by rows, the
        # engine lm_head in the lanes' order (lane_order).
        shapes["final_norm"] = ((hidden,), BF16, ld)
        shapes["lm_head"] = ((c.vocab_size, hidden), BF16, mat)
        lanes = ports * LANES_PER_PORT
        tensors, kinds, addr = {}, {}, 0
        for loaded in (False, True):  # the engine's, then LOAD's above them
            engine_bytes = addr
            for name, (shape, dtype, kind) in shapes.items():
                if (kind == ld) == loaded:
                    tensors[name], kinds[name] = (addr, shape, dtype), kind
                    addr += -(-reserved_bytes(shape, dtype, kind, lanes) // ALIGN) * ALIGN
        if addr > channel_bytes:
            raise ValueError(
                f"{c.name_or_path or 'model'} at cap {cap} needs {addr:,} bytes in channel 0,"
                f" more than its {channel_bytes:,}"
            )
        return Layout(config, cap, stride, tensors, kinds, ports, channel_bytes, engine_bytes, addr)

    def addr(self, name: str) -> int:
        return self.tensors[name][0]

    def shape(self, name: str) -> tuple[int, ...]:
        return self.tensors[name][1]

    @property
    def lanes(self) -> int:
        return self.ports * LANES_PER_PORT

    @property
    def total_bytes(self) -> int:
        """HBM reserved: the engine's region in every channel, and LOAD's
        tensors in channel 0."""
        return (self.ports - 1) * self.engine_bytes + self.end_bytes


def load(
    model: decoder.Model, layout: Layout, cache: decoder.KVCache | None = None
) -> dict[int, torch.Tensor]:
    """HBM as the host fills it: weights and RoPE tables (shared with the
    model, never written), and an empty KV cache, or a copy of `cache`'s
    filled positions (a prompt run by decoder.step, which writes the same
    bits as one decode step per position). Cache rows no token has written
    hold stale data: NaN here, so a read of one shows."""
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
        t = names[name] if name in names else torch.full(shape, torch.nan, dtype=dtype)
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
    Returns each port's beats [n_beats, LANES_PER_PORT * E], in w's dtype
    (BF16, or the tests' int32 indices, which move alike)."""
    assert w.dim() == 2 and w.dtype in (BF16, torch.int32), (w.shape, w.dtype)
    assert lanes > 0 and lanes % LANES_PER_PORT == 0, lanes
    n, m = w.shape
    ports, n_groups, n_cols = lanes // LANES_PER_PORT, -(-n // LANES_PER_PORT), -(-m // E)
    padded = torch.zeros(n_groups * LANES_PER_PORT, n_cols * E, dtype=w.dtype)
    padded[:n, :m] = w
    # Group g (rows 4g ... 4g + 3) is port g % ports's in pass g // ports.
    beats = padded.reshape(n_groups, LANES_PER_PORT, n_cols, E).transpose(1, 2)
    beats = beats.reshape(n_groups, n_cols, LANES_PER_PORT * E)
    return [beats[p::ports].reshape(-1, LANES_PER_PORT * E) for p in range(ports)]


def values_heads(d: int, lanes: int) -> int:
    """The heads a VALUES pass takes on `lanes` lanes (rtl/engine.sv's
    Heads, at most VALUES_HEADS), so the copies of V (D-062)."""
    return max(1, min(VALUES_HEADS, lanes // d))


def kv_region_beats(cap: int, d: int, values: bool, lanes: int) -> int:
    """The beats of one KV head's region in each channel (kv_lane_order):
    K's ceil(cap / lanes) tiles of ceil(d / E) beats, or V's passes (its d
    rows once per head of a pass) of ceil(cap / E)."""
    if values:
        return -(-d * values_heads(d, lanes) // lanes) * -(-cap // E)
    return -(-cap // lanes) * -(-d // E)


def region_beats(shape: tuple[int, ...], kind: Kind, lanes: int) -> int:
    """The beats a matrix or a KV cache [kv_heads, cap, d] reserves in each
    channel (D-040, D-062): the largest share, as if the last pass were
    full. The hardware needs a cache's cap a multiple of lanes."""
    if kind == Kind.MATRIX:
        n, m = shape
        return -(-n // lanes) * -(-m // E)
    kv_heads, cap, d = shape
    return kv_heads * kv_region_beats(cap, d, kind == Kind.V_CACHE, lanes)


def reserved_bytes(shape: tuple[int, ...], dtype: torch.dtype, kind: Kind, lanes: int) -> int:
    """The bytes a tensor reserves: in channel 0 if LOAD reads it, else in
    every channel."""
    if kind == Kind.LOADED:
        return n_bytes(shape, dtype)
    return region_beats(shape, kind, lanes) * BEAT_BYTES


def per_port(x: torch.Tensor, kind: Kind, lanes: int) -> list[torch.Tensor]:
    """A matrix or a KV cache in the engine's order: each port's beats
    (lane_order, kv_lane_order)."""
    if kind == Kind.MATRIX:
        return lane_order(x, lanes)
    return kv_lane_order(x, kind == Kind.V_CACHE, lanes)


def kv_lane_order(cache: torch.Tensor, values: bool, lanes: int) -> list[torch.Tensor]:
    """A KV cache [kv_heads, cap, d] in the order the engine's ports read it
    (D-062), each KV head's region after the last, the same size in every
    channel: K as a matrix of cap rows (the positions) and d columns, V
    transposed (d rows, cap columns) and stacked once per head of a VALUES
    pass (up to VALUES_HEADS: the pass's second head reads ports d / 4 ...),
    both in lane_order. A channel with no rows of a head holds zeros there.
    Returns each port's beats [n_beats, LANES_PER_PORT * E], in the cache's
    dtype (as lane_order)."""
    assert cache.dim() == 3 and cache.dtype in (BF16, torch.int32), (cache.shape, cache.dtype)
    kv_heads, cap, d = cache.shape
    ports = lanes // LANES_PER_PORT
    copies = values_heads(d, lanes) if values else 1
    region = kv_region_beats(cap, d, values, lanes)
    out = torch.zeros(ports, kv_heads, region, LANES_PER_PORT * E, dtype=cache.dtype)
    for kv in range(kv_heads):
        w = torch.cat([cache[kv].T] * copies) if values else cache[kv]
        for p, beats in enumerate(lane_order(w.contiguous(), lanes)):
            out[p, kv, : len(beats)] = beats
    return list(out.reshape(ports, kv_heads * region, LANES_PER_PORT * E))


def channel_writes(
    layout: Layout, hbm: dict[int, torch.Tensor], kinds: frozenset[Kind] = frozenset(Kind)
) -> Iterator[tuple[int, torch.Tensor]]:
    """HBM as the host writes it, one tensor at a time (of `kinds` only):
    (HBM byte address, bytes as uint8) per write. An engine tensor goes to
    every channel at its offset, in the engine's order (per_port); one LOAD
    reads goes to channel 0 as it is. Channel c starts at c · channel_bytes."""
    for name, (addr, _, _) in layout.tensors.items():
        kind = layout.kinds[name]
        if kind not in kinds:
            continue
        x = hbm[addr]
        if kind == Kind.LOADED:
            yield addr, x.contiguous().view(-1).view(torch.uint8)
            continue
        for p, beats in enumerate(per_port(x, kind, layout.lanes)):
            yield p * layout.channel_bytes + addr, beats.contiguous().view(-1).view(torch.uint8)


def port_slots(shape: tuple[int, ...], kind: Kind, lanes: int) -> list[torch.Tensor]:
    """per_port() of the element indices: per port and beat element, the
    tensor's flat index there plus one, or 0 where none is."""
    index = torch.arange(1, math.prod(shape) + 1, dtype=torch.int32).reshape(shape)
    return per_port(index, kind, lanes)


def channel_reads(
    layout: Layout,
    read: Callable[[int, int], torch.Tensor],
    kinds: frozenset[Kind] = frozenset(Kind),
) -> dict[int, torch.Tensor]:
    """HBM as channel_writes() left it, back in the tensors' logical shape
    (load()'s map; of `kinds` only): `read(address, n_bytes)` gives HBM's
    bytes as uint8. A matrix or cache is gathered from every channel's
    share."""
    hbm = {}
    for name, (addr, shape, dtype) in layout.tensors.items():
        kind = layout.kinds[name]
        if kind not in kinds:
            continue
        if kind == Kind.LOADED:
            hbm[addr] = read(addr, n_bytes(shape, dtype)).view(dtype).reshape(shape).clone()
            continue
        out, shares = torch.zeros(math.prod(shape), dtype=dtype), []
        for p, slots in enumerate(port_slots(shape, kind, layout.lanes)):
            if not len(slots):  # a port with no rows of it
                continue
            at = p * layout.channel_bytes + addr
            beats = read(at, n_bytes(tuple(slots.shape), dtype)).view(dtype).view(slots.shape)
            used = slots > 0
            index, data = slots[used].long() - 1, beats[used]
            out[index] = data
            shares.append((p, index, data))
        if kind == Kind.V_CACHE:  # its copies (D-062) agree
            for p, index, data in shares:
                same = torch.equal(out[index].view(torch.int16), data.view(torch.int16))
                assert same, f"{name}: V's copies differ on port {p}"
        hbm[addr] = out.reshape(shape)
    return hbm


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

    def kv_store(op: Op, name: str, a: Buf) -> Command:
        return Command(op, a=a, n=kv_heads, m=d, cap=layout.cap, addr=layout.addr(name))

    scale = f32_bits(host.attention_scale(d))
    residual = Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=1, m=c.hidden_size)
    return [
        *norm(layout, p + "attn_norm"),
        matvec(layout, Buf.Q, p + "q", Buf.X),
        matvec(layout, Buf.K, p + "k", Buf.X),
        matvec(layout, Buf.V, p + "v", Buf.X),
        *rope(Buf.KR, Buf.K, Buf.KR, kv_heads, d),
        kv_store(Op.KV_STORE, p + "k_cache", Buf.KR),
        kv_store(Op.V_STORE, p + "v_cache", Buf.V),
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


def input_writers(commands: list[Command]) -> list[int | None]:
    """Per command, the last command before it that wrote its input `a`, or
    None (the host filled it)."""
    last_write: dict[Buf, int] = {}
    out = []
    for i, cmd in enumerate(commands):
        out.append(last_write.get(cmd.a))
        if cmd.dst != Buf.NONE:
            last_write[cmd.dst] = i
    return out


def engine_waits(commands: list[Command]) -> list[bool]:
    """Per command, whether it is an engine command that steps only after
    its go (D-042): the last write of its input has not been waited for by
    an earlier engine command, so the controller gives the go once that
    writer is done. One that reads what an earlier engine command waited
    for (k and v after q, up after gate) does not wait, nor does one whose
    input no command wrote (the host filled it)."""
    waited: set[int] = set()  # writes some engine command waited for
    waits = []
    for cmd, writer in zip(commands, input_writers(commands), strict=True):
        wait = cmd.op in ENGINE_OPS and writer is not None and writer not in waited
        if wait:
            waited.add(writer)
        waits.append(wait)
    return waits


def stops(commands: list[Command]) -> list[bool]:
    """Per command, whether the controller stops at it until the commands
    before it are done (D-045): a command off the engine that reads a buffer
    written since the last stop. Engine commands wait inside the engine
    (engine_waits)."""
    written: set[Buf] = set()  # since the last stop
    out = []
    for cmd in commands:
        stop = cmd.op not in ENGINE_OPS and bool({cmd.a, cmd.b, cmd.c} & written)
        if stop:
            written = set()
        if cmd.dst != Buf.NONE:
            written.add(cmd.dst)
        out.append(stop)
    return out


def kv_waits(commands: list[Command]) -> list[int | None]:
    """Per command, the KV_STORE or V_STORE an engine command is given after
    (D-044): the last one before it that writes the cache it reads. None for
    the rest."""
    last_store: dict[int, int] = {}  # cache addr -> its store
    after: list[int | None] = []
    for i, cmd in enumerate(commands):
        after.append(last_store.get(cmd.addr) if cmd.op in ENGINE_OPS else None)
        if cmd.op in STORE_OPS:
            last_store[cmd.addr] = i
    return after


@dataclass(frozen=True)
class Hazard:
    """A read or write the ordering rules do not place after the access it
    must follow: `cmd` may touch `resource` before `earlier` is done."""

    kind: str  # "read after write", "write after read", "write after write", "read port"
    cmd: tuple[int, int]  # (token, index in the list)
    earlier: tuple[int, int]
    resource: Buf | tuple[int, int]  # an on-chip buffer, or (cache addr, row)


def hazards(
    commands: list[Command],
    waits: list[bool] | None = None,
    kv_after: list[int | None] | None = None,
    write_after_reads: bool = True,
) -> list[Hazard]:
    """Every access the ordering rules leave unordered, over two tokens of
    `commands` (the second at the next position). The rules (D-041, D-042,
    D-044), as the controller and the engine keep them:

    - the controller gives commands to their units in list order (handover);
    - a command that is not the engine's is handed over only after the
      last writes of the buffers it reads are done, and with
      `write_after_reads` after the earlier readers and the last writer of
      its dst are done: the controller stops there, holding every later
      command (D-045);
    - an engine command steps after the engine command before it, and a
      waiting one (`waits`) after the last write of its input is done; it
      drains in order; it reads HBM (weights, the KV cache) from its
      handover, and with `write_after_reads` is handed over only after the
      earlier readers and the last writer of its dst are done;
    - an engine command is handed over after its `kv_after` KV_STORE is done;
    - END starts after every command of its token is done, and the next
      token's first command after END.

    Nothing else is assumed: a unit may run several commands at once, and
    the engine's reads ahead may come at once. A token at position p
    writes cache row p and reads rows 0 to p.

    Besides the orderings, a buffer has one read port: two commands of
    different units (the engine; the vector unit, which also runs LOAD and
    KV_STORE) that read the same buffer at once are a "read port" hazard.
    A unit reads its commands' operands in order, so its own commands
    share a port without one. (Two writers are already ordered: "write
    after write".)"""
    waits = engine_waits(commands) if waits is None else waits
    kv_after = kv_waits(commands) if kv_after is None else kv_after
    n = len(commands)
    handover, start, done = 0, 1, 2  # each command's events, in this order

    def node(g: int, event: int) -> int:
        """Event of command g (global: token * n + index)."""
        return 3 * g + event

    preds: dict[int, list[int]] = {}

    def edge(a: int, b: int) -> None:
        preds.setdefault(b, []).append(a)

    Resource = Buf | tuple[int, int]
    readers: dict[Resource, list[int]] = {}  # since the last write
    writer: dict[Resource, int] = {}
    checks: list[tuple[str, int, int, Resource, int]] = []  # kind, cmd, earlier, what, at node
    last_engine: int | None = None

    def is_engine(g: int) -> bool:
        """Whether command g is the engine's (else the vector unit's)."""
        return commands[g % n].op in ENGINE_OPS

    for token in range(2):
        for i, cmd in enumerate(commands):
            g = token * n + i
            h, st, d = node(g, handover), node(g, start), node(g, done)
            edge(h, st)
            edge(st, d)
            if g:
                edge(node(g - 1, handover), h)
            engine = cmd.op in ENGINE_OPS
            operands = dict.fromkeys(b for b in (cmd.a, cmd.b, cmd.c) if b != Buf.NONE)
            reads: list[tuple[Resource, int]] = [(b, st) for b in operands]
            for b, _ in reads:
                if b in writer and (not engine or waits[i]):
                    edge(node(writer[b], done), st if engine else h)
            if cmd.op in (Op.SCORES, Op.VALUES):  # from the handover: read ahead
                reads += [((cmd.addr, row), h) for row in range(token + 1)]
            if engine:
                if last_engine is not None:
                    edge(node(last_engine, start), st)
                    edge(node(last_engine, done), d)
                last_engine = g
                if kv_after[i] is not None:
                    edge(node(token * n + kv_after[i], done), h)
            if cmd.op == Op.END:
                for j in range(i):
                    edge(node(token * n + j, done), h)
            writes: list[Resource] = [cmd.dst] if cmd.dst != Buf.NONE else []
            if cmd.op in STORE_OPS:
                writes.append((cmd.addr, token))
            for what, at in reads:
                if what in writer:
                    checks.append(("read after write", g, writer[what], what, at))
                if isinstance(what, Buf):
                    others = (k for k in readers.get(what, []) if is_engine(k) != engine)
                    checks += [("read port", g, k, what, at) for k in others]
                readers.setdefault(what, []).append(g)
            for what in writes:
                earlier = [("write after read", k) for k in readers.get(what, []) if k != g]
                if what in writer:
                    earlier.append(("write after write", writer[what]))
                for kind, k in earlier:
                    checks.append((kind, g, k, what, st))
                    if write_after_reads and isinstance(what, Buf):
                        edge(node(k, done), h)
                writer[what], readers[what] = g, []

    # ancestors[v]: the events placed before v, as a bit set
    ancestors = [0] * node(2 * n, 0)
    for v in range(len(ancestors)):
        for u in preds.get(v, []):
            ancestors[v] |= ancestors[u] | (1 << u)
    return [
        Hazard(kind, divmod(g, n), divmod(k, n), what)
        for kind, g, k, what, at in checks
        if not ancestors[at] >> node(k, done) & 1
    ]


@dataclass(frozen=True)
class EngineCall:
    """One engine command as run() executed it: what the engine reads and
    what it writes, bit for bit (the RTL tests' real data)."""

    index: int  # in the command list
    name: str  # the Layout name of what it reads in HBM
    cmd: Command
    t: int  # the positions attention reads
    stride: int  # the STRIDE register: S and P's rows
    a: torch.Tensor  # what it reads of its input buffer, BF16 (engine_spans)
    mem: torch.Tensor  # the matrix, or a copy of the cache (rows at or past t stale)
    out: torch.Tensor  # what it writes of dst: FP32, or rounded where the buffer is BF16
    wait: bool  # steps only after its go (engine_waits)

    def rows(self, x: torch.Tensor) -> torch.Tensor:
        """x (S or P as recorded: n rows at the stride) as its n rows of t."""
        return x.view(self.cmd.n, self.stride)[:, : self.t]


def engine_spans(cmd: Command, stride: int) -> tuple[int, int]:
    """The elements an engine command reads from the start of its input
    buffer and writes from the start of dst: SCORES writes rows of t at
    `stride` (past t, the rows keep what they held), VALUES reads them."""
    match cmd.op:
        case Op.MATVEC:
            return cmd.m, cmd.n
        case Op.SCORES:
            return cmd.n * cmd.m, cmd.n * stride
        case Op.VALUES:
            return cmd.n * stride, cmd.n * cmd.m
    raise AssertionError(f"not an engine command: {cmd.op!r}")


VECTOR_OPS = frozenset(OPS) - ENGINE_OPS - STORE_OPS - {Op.LOAD, Op.OUTPUT, Op.END}
UNARY = {Op.EXP: funcs.exp, Op.RECIP: funcs.recip, Op.RSQRT: funcs.rsqrt}
BINARY = {Op.ADD: arith.add, Op.MUL: arith.mul}
REDUCE = {Op.SUM: vector.reduce_sum, Op.SUMSQ: vector.reduce_sum_squares, Op.MAX: vector.reduce_max}


def empty_buffers(formats: dict[Buf, tuple[torch.dtype, int]]) -> dict[Buf, torch.Tensor]:
    """The on-chip buffers at the start of a token: stale data, NaN here, so
    a read of an element no command wrote shows."""
    return {
        b: torch.full((length,), torch.nan, dtype=dtype) for b, (dtype, length) in formats.items()
    }


def buffer_span(buf: torch.Tensor, n: int, length: int, row_stride: int = 0) -> torch.Tensor:
    """n rows of `length` elements of a buffer, `row_stride` apart (packed by
    default), as a view."""
    row_stride = row_stride or length
    assert length <= row_stride and n * row_stride <= len(buf), (len(buf), n, length, row_stride)
    return buf[: n * row_stride].view(n, row_stride)[:, :length]


def store(dst: torch.Tensor, x: torch.Tensor) -> None:
    """Write an FP32 result into a span, rounded if the buffer is BF16."""
    assert x.dtype == F32 and x.shape == dst.shape, (x.dtype, x.shape, dst.shape)
    dst.copy_(arith.bf16(x) if dst.dtype == BF16 else x)


def overlap_ok(cmd: Command) -> bool:
    """Whether a command's buffers may be shared as it uses them. Its
    sources are distinct buffers: a buffer has one read port, so two
    operands of a group cannot read it at different elements. dst may be a
    source only where an element is read once, before its own write (the
    vector unit writes a group a pipeline's latency after reading it): not
    ROTATE_HALF's a, nor a b read for every row (B_ROW) or by row
    (B_PER_ROW)."""
    sources = [x for x in (cmd.a, cmd.b, cmd.c) if x != Buf.NONE]
    if len(set(sources)) < len(sources):
        return False
    if cmd.op == Op.ROTATE_HALF and cmd.dst == cmd.a:
        return False
    return not (cmd.flags & (Flag.B_ROW | Flag.B_PER_ROW) and cmd.dst == cmd.b)


def vector_command(cmd: Command, buf: dict[Buf, torch.Tensor], t: int, stride: int) -> None:
    """Run one vector-unit command (VECTOR_OPS) on the buffers, in place: t
    is the positions attention reads, stride the STRIDE register. Its
    sources are FP32 buffers."""
    assert cmd.op in VECTOR_OPS and flags_ok(cmd) and overlap_ok(cmd), cmd
    length = cmd.row_length(t)
    assert cmd.n >= 1 and length >= 1, f"no rows or no elements: {cmd}"  # the unit needs both
    assert cmd.op != Op.ROTATE_HALF or length % 2 == 0, f"odd rows to rotate: {cmd}"
    sources = [x for x in (cmd.a, cmd.b, cmd.c) if x != Buf.NONE]
    assert all(buf[x].dtype == F32 for x in sources), cmd

    def rows(x: Buf) -> torch.Tensor:
        """Buffer x as the command's n rows: of m, or of t at the stride."""
        if cmd.flags & Flag.LEN_T:
            return buffer_span(buf[x], cmd.n, t, stride)
        return buffer_span(buf[x], cmd.n, cmd.m)

    def operand_a() -> torch.Tensor:
        v = rows(cmd.a)
        return -v if cmd.flags & Flag.NEG_A else v  # unary minus: an exact sign flip

    def operand_b() -> torch.Tensor:
        if cmd.b == Buf.NONE:
            v = f32_value(cmd.scalar)
        elif cmd.flags & Flag.B_PER_ROW:
            v = buffer_span(buf[cmd.b], cmd.n, 1)
        elif cmd.flags & Flag.B_ROW:
            v = buffer_span(buf[cmd.b], 1, cmd.row_length(t))
        else:
            v = rows(cmd.b)
        return -v if cmd.flags & Flag.NEG_B else v

    match cmd.op:
        case Op.ADD | Op.MUL:
            store(rows(cmd.dst), BINARY[cmd.op](operand_a(), operand_b()))
        case Op.FMA:
            store(rows(cmd.dst), arith.fma(operand_a(), operand_b(), rows(cmd.c)))
        case Op.EXP | Op.RECIP | Op.RSQRT:
            store(rows(cmd.dst), UNARY[cmd.op](operand_a()))
        case Op.SUM | Op.SUMSQ | Op.MAX:
            store(buffer_span(buf[cmd.dst], 1, cmd.n), REDUCE[cmd.op](rows(cmd.a))[None])
        case Op.ROTATE_HALF:
            store(rows(cmd.dst), vector.rotate_half(rows(cmd.a)))


def load_command(
    cmd: Command, buf: dict[Buf, torch.Tensor], mem: torch.Tensor, token: int, position: int
) -> None:
    """LOAD: dst = up(the tensor `mem` at cmd.addr, or its row `token` or
    `position`), m elements."""
    assert (mem.dtype == BF16) == bool(cmd.flags & Flag.SRC_BF16), (mem.dtype, cmd)
    if cmd.flags & Flag.BY_TOKEN:
        mem = mem[token]
    elif cmd.flags & Flag.BY_POSITION:
        mem = mem[position]
    assert mem.numel() == cmd.m, (mem.shape, cmd.m)
    mem = mem.reshape(1, -1)
    store(buffer_span(buf[cmd.dst], 1, cmd.m), arith.up(mem) if mem.dtype == BF16 else mem)


def kv_store_command(
    cmd: Command, buf: dict[Buf, torch.Tensor], mem: torch.Tensor, position: int
) -> None:
    """KV_STORE or V_STORE: row `position` of each of the n heads of the
    cache `mem` (n × cap × m, BF16) = a's n rows of m, rounded. m (the head
    dimension) is a power of two. In HBM the cache is in the engine's order
    (kv_lane_order, D-062)."""
    assert mem.shape == (cmd.n, cmd.cap, cmd.m) and buf[cmd.a].dtype == F32, (mem.shape, cmd)
    assert cmd.m & (cmd.m - 1) == 0, cmd
    mem[:, position] = arith.bf16(buffer_span(buf[cmd.a], cmd.n, cmd.m))


def validate(layout: Layout, cmds: list[Command], buffer_elements: int | None = None) -> None:
    """Raise ValueError on the first command the hardware cannot run
    correctly, or on a layout it cannot hold (with buffers of
    `buffer_elements` elements each; None: no limit): the controller checks
    only for unknown opcodes (PLAN.md, M7's answers), so the runtime calls
    this before writing a list. Ordering hazards included: the controller's
    waits and stops (D-068) do not keep two commands off one read port."""
    lanes = layout.lanes

    def need(ok: bool, i: int, why: str) -> None:
        if not ok:
            raise ValueError(f"command {i} ({cmds[i].op!r}): {why}")

    most = COMMAND_BUFFER_BYTES // COMMAND_BYTES
    if not 0 < len(cmds) <= most:
        raise ValueError(f"{len(cmds)} commands: the buffer holds 1 to {most}")
    if layout.stride % STRIDE_ALIGN or not 0 < layout.stride < 2**ROW_BITS:
        raise ValueError(f"stride {layout.stride}: a multiple of {STRIDE_ALIGN} below 2^{ROW_BITS}")
    if layout.end_bytes > layout.channel_bytes:
        raise ValueError(
            f"layout: {layout.end_bytes:,} bytes, channel 0 holds {layout.channel_bytes:,}"
        )
    longest = max(n for _, n in buffers(layout.config, layout.stride).values())
    if buffer_elements is not None and longest > buffer_elements:
        raise ValueError(
            f"buffers of {longest:,} elements needed, the device has {buffer_elements:,}"
        )
    for i, cmd in enumerate(cmds):
        need(cmd.op in OPS, i, "unknown opcode")
        need((cmd.op == Op.END) == (i == len(cmds) - 1), i, "END must be the last command, once")
        need(flags_ok(cmd), i, f"flags {cmd.flags!r} not accepted")
        need(overlap_ok(cmd), i, "dst overlaps a source it reads again")
        need(max(cmd.n, cmd.m, cmd.cap) < 2**ROW_BITS, i, f"n, m or cap of {ROW_BITS} bits or more")
        if cmd.op in (Op.END, Op.OUTPUT):
            continue
        need(cmd.m >= 1 or bool(cmd.flags & Flag.LEN_T), i, "no columns (m = 0)")  # LEN_T: t
        if cmd.op == Op.LOAD:  # rows of whole AXI beats (rtl/vec_unit.sv)
            beat = VECTOR_LANES if cmd.flags & Flag.SRC_BF16 else VECTOR_LANES // 2
            need(cmd.m % beat == 0, i, f"m {cmd.m} not whole beats ({beat} elements)")
            need(cmd.addr % BEAT_BYTES == 0, i, "address not beat-aligned")
        else:
            need(cmd.n >= 1, i, "no rows (n = 0)")
        if cmd.op in STORE_OPS:
            need(cmd.addr % CACHE_ALIGN == 0, i, f"cache address not {CACHE_ALIGN}-byte aligned")
        if cmd.op == Op.ROTATE_HALF:
            need(cmd.m % 2 == 0, i, "odd row length")
        if cmd.op in ENGINE_OPS:
            need(cmd.m < 2**COL_BITS, i, f"m of {COL_BITS} bits or more")
            need(cmd.addr % BEAT_BYTES == 0, i, "address not beat-aligned")
        if cmd.op in ENGINE_OPS - {Op.MATVEC} | STORE_OPS:
            need(cmd.cap % lanes == 0, i, f"cap {cmd.cap} not a multiple of the {lanes} lanes")
            need(cmd.m in HEAD_DIMS, i, f"head dimension {cmd.m} not one of {HEAD_DIMS}")
        if cmd.op in (Op.VALUES, Op.V_STORE):
            need(cmd.m <= lanes, i, f"head dimension {cmd.m} above the {lanes} lanes")
        if cmd.op in (Op.SCORES, Op.VALUES):
            need(0 < cmd.kv_heads and cmd.n % cmd.kv_heads == 0, i, "KV heads not dividing heads")
            need(cmd.n < 2**HEAD_BITS, i, f"heads of {HEAD_BITS} bits or more")
    found = hazards(cmds)
    if found:
        raise ValueError(f"{len(found)} ordering hazards, first {found[0]}")


def run(
    commands: list[Command],
    layout: Layout,
    hbm: dict[int, torch.Tensor],
    token: int,
    position: int,
    record: Callable[[EngineCall], None] | None = None,
    start: dict[Buf, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Execute a command list for one token, as the controller does: buffers
    start with stale data (empty_buffers, or `start`, updated in place),
    HBM (the KV cache) is updated in place. Returns what OUTPUT wrote to the
    host, or None for a list without OUTPUT (a prompt token whose logits the
    host does not need). `record` gets every engine command's EngineCall."""
    assert len(commands) * COMMAND_BYTES <= COMMAND_BUFFER_BYTES, len(commands)
    names = {addr: name for name, (addr, _, _) in layout.tensors.items()} if record else {}
    waits = engine_waits(commands) if record else []
    formats = buffers(layout.config, layout.stride)
    buf = empty_buffers(formats) if start is None else start
    assert all(buf[b].dtype == dt and len(buf[b]) >= n for b, (dt, n) in formats.items()), formats
    output = None
    t = position + 1  # positions attention reads: 0 ... position
    stride = layout.stride  # the STRIDE register
    assert t <= min(layout.cap, stride), (t, layout.cap, stride)

    def span(x: Buf, n: int, length: int, row_stride: int = 0) -> torch.Tensor:
        return buffer_span(buf[x], n, length, row_stride)

    for i, cmd in enumerate(commands):
        assert cmd.op in OPS, f"unknown command {cmd.op!r}"  # stops, as case _ below
        assert flags_ok(cmd), f"command {i}: flags {cmd.flags!r} not accepted by {cmd.op!r}"
        assert overlap_ok(cmd), f"command {i}: dst {cmd.dst!r} overlaps a source it reads again"
        a, mem = buf.get(cmd.a), hbm.get(cmd.addr)
        match cmd.op:
            case Op.LOAD:
                load_command(cmd, buf, mem, token, position)
            case Op.MATVEC:
                assert mem.shape == (cmd.n, cmd.m) and a.dtype == BF16, (mem.shape, a.dtype)
                store(span(cmd.dst, 1, cmd.n), dot.matvec(mem, span(cmd.a, 1, cmd.m)))  # one vector
            case Op.KV_STORE | Op.V_STORE:
                kv_store_command(cmd, buf, mem, position)
            case Op.SCORES:
                q = span(cmd.a, cmd.n, cmd.m).reshape(cmd.kv_heads, -1, cmd.m)  # KV head h // group
                s = decoder.attention_scores(q, mem[:, None, :t], f32_value(cmd.scalar))
                store(span(cmd.dst, cmd.n, t, stride), s.reshape(cmd.n, t))
            case Op.VALUES:
                p = span(cmd.a, cmd.n, t, stride).reshape(cmd.kv_heads, -1, t)
                o = decoder.attention_values(p, mem[:, None, :t])
                store(span(cmd.dst, cmd.n, cmd.m), o.reshape(cmd.n, cmd.m))
            case Op.OUTPUT:
                assert a.dtype == F32, a.dtype
                output = span(cmd.a, 1, cmd.n)[0].clone()
            case op if op in VECTOR_OPS:
                vector_command(cmd, buf, t, stride)
            case Op.END:  # must be the last command (the COMMANDS register's count)
                assert i == len(commands) - 1, f"{len(commands) - 1 - i} commands after END"
                return output
            case _:  # the controller stops with STATUS error and the index in ERROR
                raise AssertionError(f"unknown command {cmd.op!r}")
        if record is not None and cmd.op in ENGINE_OPS:
            read = mem if cmd.op == Op.MATVEC else mem.clone()  # later tokens write the cache
            n_in, n_out = engine_spans(cmd, stride)
            x, out = span(cmd.a, 1, n_in)[0].clone(), span(cmd.dst, 1, n_out)[0].clone()
            record(EngineCall(i, names[cmd.addr], cmd, t, stride, x, read, out, waits[i]))
    raise AssertionError("command list without END")


def record_step(
    model: decoder.Model,
    prompt: list[int],
    token: int,
    cache: decoder.KVCache | None = None,
    cap: int | None = None,
    stride: int | None = None,
) -> tuple[torch.Tensor, list[EngineCall]]:
    """One decode step of `token` at position len(prompt), after `prompt`
    (run by decoder.step, or given as `cache`, its KV cache copied into HBM):
    the logits and the step's engine commands as run() executed them. The
    KV cache holds `cap` positions (by default just the step's), S and P's
    rows sit at `stride` (by default cap)."""
    layout = Layout.of(model.config, len(prompt) + 1 if cap is None else cap, stride)
    if cache is None:
        cache = decoder.KVCache.empty(model, layout.cap)
        if prompt:
            decoder.step(model, cache, prompt)
    assert cache.length == len(prompt), (cache.length, len(prompt))
    hbm, calls = load(model, layout, cache), []
    logits = run(build(layout), layout, hbm, token, len(prompt), calls.append)
    assert logits is not None
    return logits, calls
