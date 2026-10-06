# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""The command list of one decode step (docs/architecture.md, "Commands").

The host builds the list once per model (build()) and writes it into the
controller's command buffer; the controller runs it for every token, with the
token id and the position in registers, so the list does not change between
tokens. run() executes a list with the golden model's functions: it is the
reference for the controller, and the tests check that it gives the golden
decode step's bits, so every operation of the step has a command.

Addresses are HBM byte addresses from a Layout. HBM is modelled as a map from
address to tensor, in the tensors' logical shape: the engine's read order
(docs/architecture.md, "Weight layout") is a reordering the model leaves out.
"""

import enum
import struct
from dataclasses import dataclass

import torch
from golden import arith, decoder, dot, host, vector
from transformers import LlamaConfig

ALIGN = 4096  # HBM bytes; every tensor starts on a boundary


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


# What each command does, on which fields (docs/architecture.md, "Commands").
OP_DOC = {
    Op.END: "token done: set the status register's done bit, raise the interrupt",
    Op.EMBED: "dst = up(table row `token`); table at addr, rows of m BF16",
    Op.RMSNORM: "dst = bf16(rmsnorm(a, gain at addr, eps = scalar)), n elements",
    Op.MATVEC: "dst = W·a, W at addr: n rows × m cols BF16; a is BF16, dst FP32",
    Op.ROPE: "dst = bf16(rope(a)): n vectors of m; cos and sin rows at addr, row `position`",
    Op.KV_STORE: "cache[h][position] = a[h] (rounded to BF16 if FP32), n heads of m; cache at addr",
    Op.SCORES: "dst[h][t] = dot(a[h], K[h // group][t]) · scalar for t ≤ position",
    Op.SOFTMAX: "dst = bf16(softmax(a)), n vectors of position + 1",
    Op.VALUES: "dst[h] = bf16(Σₜ a[h][t] · V[h // group][t]) for t ≤ position",
    Op.SWIGLU: "dst = bf16(swiglu(a, b)), n elements",
    Op.ADD: "dst = add(a, b), n elements",
    Op.OUTPUT: "write a (n FP32) to host memory at the logits address register",
}


# Commands that read or write HBM at `addr`.
ADDRESSED = {Op.EMBED, Op.RMSNORM, Op.MATVEC, Op.ROPE, Op.KV_STORE, Op.SCORES, Op.VALUES}


class Buf(enum.IntEnum):
    """On-chip buffers, named by what they hold."""

    NONE = 0
    H = 1  # residual stream, FP32 [hidden]
    X = 2  # normalized input of a matrix product, BF16 [hidden]
    Q = 3  # FP32 [heads × head_dim]
    K = 4  # FP32 [kv_heads × head_dim]
    V = 5  # FP32 [kv_heads × head_dim]
    QR = 6  # q after RoPE, BF16 [heads × head_dim]
    KR = 7  # k after RoPE, BF16 [kv_heads × head_dim]
    S = 8  # scores, FP32 [heads × positions]
    P = 9  # probabilities, BF16 [heads × positions]
    ATT = 10  # attention output, BF16 [heads × head_dim]
    T = 11  # a product to add to the residual stream, FP32 [hidden]
    G = 12  # gate, FP32 [intermediate]
    U = 13  # up, FP32 [intermediate]
    M = 14  # SwiGLU output, BF16 [intermediate]
    LOGITS = 15  # FP32 [vocab]


def buffer_bytes(config: LlamaConfig, cap: int) -> dict[Buf, int]:
    """Each buffer's size for a model and a cache of `cap` positions."""
    c = config
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    hidden, inter, heads = c.hidden_size, c.intermediate_size, c.num_attention_heads
    elems = {
        Buf.H: (hidden, 4),
        Buf.X: (hidden, 2),
        Buf.Q: (q, 4),
        Buf.K: (kv, 4),
        Buf.V: (kv, 4),
        Buf.QR: (q, 2),
        Buf.KR: (kv, 2),
        Buf.S: (heads * cap, 4),
        Buf.P: (heads * cap, 2),
        Buf.ATT: (q, 2),
        Buf.T: (hidden, 4),
        Buf.G: (inter, 4),
        Buf.U: (inter, 4),
        Buf.M: (inter, 2),
        Buf.LOGITS: (c.vocab_size, 4),
    }
    return {b: n * size for b, (n, size) in elems.items()}


# The 32-byte encoding: field, struct format, meaning. Little-endian.
FIELDS = [
    ("op", "B", "opcode (Op)"),
    ("dst", "B", "destination buffer (Buf)"),
    ("a", "B", "first source buffer"),
    ("b", "B", "second source buffer"),
    ("n", "I", "rows, elements or vectors"),
    ("m", "I", "columns or vector length"),
    ("heads", "H", "query heads (SCORES, VALUES)"),
    ("kv_heads", "H", "KV heads (SCORES, VALUES)"),
    ("cap", "I", "KV cache rows per head (KV_STORE, SCORES, VALUES)"),
    ("addr", "Q", "HBM byte address"),
    ("scalar", "I", "FP32 bits: eps (RMSNORM) or scale (SCORES)"),
]
FORMAT = "<" + "".join(f for _, f, _ in FIELDS)
COMMAND_BYTES = struct.calcsize(FORMAT)

# Host-visible registers on the shell's OCL port: offset, name, access, meaning.
REGISTERS = [
    (0x00, "ID", "R", "0x43424F53 ('CBOS'), then the version in the next word"),
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
    (0x10000, "COMMAND_BUFFER", "W", "the command list, COMMAND_BYTES per command"),
]


@dataclass(frozen=True)
class Command:
    op: Op
    dst: Buf = Buf.NONE
    a: Buf = Buf.NONE
    b: Buf = Buf.NONE
    n: int = 0
    m: int = 0
    heads: int = 0
    kv_heads: int = 0
    cap: int = 0
    addr: int = 0
    scalar: int = 0

    def encode(self) -> bytes:
        return struct.pack(FORMAT, *(getattr(self, name) for name, _, _ in FIELDS))

    @staticmethod
    def decode(raw: bytes) -> Command:
        values = dict(zip((name for name, _, _ in FIELDS), struct.unpack(FORMAT, raw), strict=True))
        return Command(
            **values | {"op": Op(values["op"])} | {k: Buf(values[k]) for k in ("dst", "a", "b")}
        )


def f32_bits(x: float | torch.Tensor) -> int:
    """The FP32 bits of x (a float is rounded to FP32 once)."""
    return int(arith.bits_f32(torch.as_tensor(x, dtype=torch.float32).reshape(1))[0])


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
        bf16, f32 = torch.bfloat16, torch.float32
        shapes = {"embed": ((c.vocab_size, hidden), bf16), "rope": ((cap, 2, d), f32)}
        for i in range(c.num_hidden_layers):
            p = f"layers.{i}."
            shapes |= {
                p + "attn_norm": ((hidden,), bf16),
                p + "q": ((q, hidden), bf16),
                p + "k": ((kv, hidden), bf16),
                p + "v": ((kv, hidden), bf16),
                p + "o": ((hidden, q), bf16),
                p + "mlp_norm": ((hidden,), bf16),
                p + "gate": ((inter, hidden), bf16),
                p + "up": ((inter, hidden), bf16),
                p + "down": ((hidden, inter), bf16),
                p + "k_cache": ((c.num_key_value_heads, cap, d), bf16),
                p + "v_cache": ((c.num_key_value_heads, cap, d), bf16),
            }
        shapes["final_norm"] = ((hidden,), bf16)
        if not c.tie_word_embeddings:
            shapes["lm_head"] = ((c.vocab_size, hidden), bf16)
        tensors, addr = {}, 0
        for name, (shape, dtype) in shapes.items():
            tensors[name] = (addr, shape, dtype)
            size = torch.Size(shape).numel() * dtype.itemsize
            addr += -(-size // ALIGN) * ALIGN
        if c.tie_word_embeddings:
            tensors["lm_head"] = tensors["embed"]
        return Layout(config, cap, tensors)

    def addr(self, name: str) -> int:
        return self.tensors[name][0]

    @property
    def total_bytes(self) -> int:
        return max(a + torch.Size(s).numel() * t.itemsize for a, s, t in self.tensors.values())


def load(model: decoder.Model, layout: Layout) -> dict[int, torch.Tensor]:
    """HBM as the host fills it: weights, RoPE tables, an empty KV cache."""
    names = {
        "embed": model.embed,
        "rope": torch.stack([model.cos, model.sin], dim=1)[: layout.cap],
        "final_norm": model.final_norm,
        "lm_head": model.lm_head,
    }
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}."
        names |= {p + f: getattr(layer, f) for f in decoder.Layer.__dataclass_fields__}
    hbm = {}
    for name, (addr, shape, dtype) in layout.tensors.items():
        t = names[name] if name in names else torch.zeros(shape, dtype=dtype)
        assert t.shape == shape and t.dtype == dtype, (name, t.shape, shape, t.dtype)
        hbm[addr] = t.clone()
    return hbm


def build(layout: Layout) -> list[Command]:
    """One decode step, in the order of golden.decoder.step()."""
    c = layout.config
    d, hidden, inter = c.head_dim, c.hidden_size, c.intermediate_size
    heads, kv_heads = c.num_attention_heads, c.num_key_value_heads
    q_dim, kv_dim = heads * d, kv_heads * d
    eps = f32_bits(c.rms_norm_eps)
    scale = f32_bits(host.attention_scale(d))
    attention = dict(heads=heads, kv_heads=kv_heads, m=d, cap=layout.cap)
    rope = layout.addr("rope")

    def matvec(dst: Buf, name: str, rows: int, cols: int, a: Buf) -> Command:
        return Command(Op.MATVEC, dst, a, n=rows, m=cols, addr=layout.addr(name))

    out = [Command(Op.EMBED, Buf.H, m=hidden, addr=layout.addr("embed"))]
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        out += [
            Command(
                Op.RMSNORM, Buf.X, Buf.H, n=hidden, addr=layout.addr(p + "attn_norm"), scalar=eps
            ),
            matvec(Buf.Q, p + "q", q_dim, hidden, Buf.X),
            matvec(Buf.K, p + "k", kv_dim, hidden, Buf.X),
            matvec(Buf.V, p + "v", kv_dim, hidden, Buf.X),
            Command(Op.ROPE, Buf.KR, Buf.K, n=kv_heads, m=d, addr=rope),
            Command(
                Op.KV_STORE,
                a=Buf.KR,
                n=kv_heads,
                m=d,
                cap=layout.cap,
                addr=layout.addr(p + "k_cache"),
            ),
            Command(
                Op.KV_STORE,
                a=Buf.V,
                n=kv_heads,
                m=d,
                cap=layout.cap,
                addr=layout.addr(p + "v_cache"),
            ),
            Command(Op.ROPE, Buf.QR, Buf.Q, n=heads, m=d, addr=rope),
            Command(
                Op.SCORES, Buf.S, Buf.QR, addr=layout.addr(p + "k_cache"), scalar=scale, **attention
            ),
            Command(Op.SOFTMAX, Buf.P, Buf.S, n=heads, cap=layout.cap),
            Command(Op.VALUES, Buf.ATT, Buf.P, addr=layout.addr(p + "v_cache"), **attention),
            matvec(Buf.T, p + "o", hidden, q_dim, Buf.ATT),
            Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=hidden),
            Command(
                Op.RMSNORM, Buf.X, Buf.H, n=hidden, addr=layout.addr(p + "mlp_norm"), scalar=eps
            ),
            matvec(Buf.G, p + "gate", inter, hidden, Buf.X),
            matvec(Buf.U, p + "up", inter, hidden, Buf.X),
            Command(Op.SWIGLU, Buf.M, Buf.G, Buf.U, n=inter),
            matvec(Buf.T, p + "down", hidden, inter, Buf.M),
            Command(Op.ADD, Buf.H, Buf.H, Buf.T, n=hidden),
        ]
    return out + [
        Command(Op.RMSNORM, Buf.X, Buf.H, n=hidden, addr=layout.addr("final_norm"), scalar=eps),
        matvec(Buf.LOGITS, "lm_head", c.vocab_size, hidden, Buf.X),
        Command(Op.OUTPUT, a=Buf.LOGITS, n=c.vocab_size),
        Command(Op.END),
    ]


def run(
    commands: list[Command], hbm: dict[int, torch.Tensor], token: int, position: int
) -> torch.Tensor:
    """Execute a command list for one token, as the controller does: buffers
    start empty, HBM (the KV cache) is updated in place. Returns what OUTPUT
    wrote to the host."""
    buf: dict[Buf, torch.Tensor] = {}
    output = None
    t = position + 1  # positions attention reads: 0 ... position

    def heads_of(x: torch.Tensor, cmd: Command, last: int) -> torch.Tensor:
        group = cmd.heads // cmd.kv_heads  # query head h uses KV head h // group
        return x.reshape(cmd.kv_heads, group, last)

    for cmd in commands:
        a, b, mem = buf.get(cmd.a), buf.get(cmd.b), hbm.get(cmd.addr)
        match cmd.op:
            case Op.EMBED:
                assert mem.shape[1] == cmd.m, (mem.shape, cmd.m)
                buf[cmd.dst] = arith.up(mem[token])
            case Op.RMSNORM:
                eps = float(f32_value(cmd.scalar))
                buf[cmd.dst] = arith.bf16(vector.rmsnorm(a, mem, eps))
            case Op.MATVEC:
                assert mem.shape == (cmd.n, cmd.m) and a.dtype == torch.bfloat16, (
                    mem.shape,
                    a.dtype,
                )
                buf[cmd.dst] = dot.matvec(mem, a[None])[0]  # one vector
            case Op.ROPE:
                cos, sin = mem[position]
                buf[cmd.dst] = arith.bf16(vector.rope(a.reshape(cmd.n, cmd.m), cos, sin)).reshape(
                    -1
                )
            case Op.KV_STORE:
                assert mem.shape == (cmd.n, cmd.cap, cmd.m), (mem.shape, cmd)
                x = a.reshape(cmd.n, cmd.m)
                mem[:, position] = x if x.dtype == torch.bfloat16 else arith.bf16(x)
            case Op.SCORES:
                keys = mem[:, None, :t]  # [kv_heads, 1, t, d]
                s = decoder.attention_scores(heads_of(a, cmd, cmd.m), keys, f32_value(cmd.scalar))
                buf[cmd.dst] = s.reshape(cmd.heads, t)
            case Op.SOFTMAX:
                buf[cmd.dst] = arith.bf16(vector.softmax(a.reshape(cmd.n, t)))
            case Op.VALUES:
                values = mem[:, None, :t]  # [kv_heads, 1, t, d]
                o = decoder.attention_values(heads_of(a, cmd, t), values)
                buf[cmd.dst] = arith.bf16(o).reshape(-1)
            case Op.SWIGLU:
                buf[cmd.dst] = arith.bf16(vector.swiglu(a, b))
            case Op.ADD:
                buf[cmd.dst] = arith.add(a, b)
            case Op.OUTPUT:
                assert a.shape == (cmd.n,) and a.dtype == torch.float32, (a.shape, a.dtype)
                output = a.clone()
            case Op.END:
                return output
    raise AssertionError("command list without END")
