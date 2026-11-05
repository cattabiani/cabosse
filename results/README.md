# Cabosse: results

Cabosse is an open-source LLM inference accelerator. This page holds the
measured results, one row per build, the first build being the baseline to
improve on. Every number comes from a saved log in
[reports/data/f2/](../reports/data/f2/); estimates are marked as such.

## A chat on the FPGA

![SmolLM2-135M-Instruct answering two questions on the Cabosse core on AWS F2, at the recorded speed](chat-f2.svg)

SmolLM2-135M-Instruct on the Cabosse core, on one AWS F2 FPGA, played back
at the recorded speed (the animation loops). The model is the official
checkpoint, BF16, with no conversion, quantization or calibration. Both
answers are greedy and stop at the end of their turn. The second question
carries the first exchange in the KV cache on the card, so only its 14 new
tokens are fed. Recording:
[chat.cast](../reports/data/f2/2026-11-05-core-run2/chat/chat.cast)
(asciicast v2), [transcript](../reports/data/f2/2026-11-05-core-run2/chat/chat.txt),
[log](../reports/data/f2/2026-11-05-core-run2/chat/chat.jsonl).

## Baseline: core build 1 (2026-11-05)

**Correct:** every generated token and every bit of every logit equals the
project's golden model (the bit-exact reference in PyTorch): three 64-token
prompts and one 1000-token generation, in each of two F2 sessions.

| | first 64 tokens | positions 46–1045 |
|---|---|---|
| tokens/s, the core alone (its cycle counter at 100 MHz) | 194.5 | 173.8 |
| cycles per token (median) | 514,027 | 575,374 |
| tokens/s, end to end on the host | 68.0 | 65.4 |

"End to end" is the host's wall time per token. It includes reading the
196 KB of logits back over PCIe at about 22 MB/s, which takes most of the
time today; the core's own time is about 5 ms a token. In the chat, the
median is 13.9 ms a token.

| | |
|---|---|
| FPGA | AWS F2 (f2.6xlarge), AMD Virtex UltraScale+ VU47P with HBM |
| Core clock | 100 MHz. Designed for 250 MHz; this build's timing allows about 105 MHz (worst slack −5.466 ns at a 4 ns target) |
| Resources | LUTs 62.2%, registers 22.0%, UltraRAM 26.7%, block RAM 10.5%, DSPs 2.7% |
| Numerics | BF16 × BF16 multiplies, FP32 accumulation, FP32 vector unit |
| Build | Vivado 2025.2, 6.6 h on a c6i.8xlarge |
| Host ↔ card copies over PCIe | write 378.8 MB/s, read 22.4 MB/s (medians of 5) |

Logs: [build](../reports/data/f2/2026-11-04-core1.txt),
[run 1](../reports/data/f2/2026-11-05-core-run1.txt),
[run 2](../reports/data/f2/2026-11-05-core-run2/core.jsonl).

**What limits it now:** the clock (the slowest path is a read from the
vector unit's on-chip buffers into its arithmetic chain), and the logits'
trip back to the host. The project's earlier starting point, Gemmini +
Rocket on the same F2, ran a 260K-parameter model at 9.0 tokens/s at
29.8 MHz; this model is about 500 times larger.

### Against the same F2 host's CPU

Same session, same model, same machine: the f2.6xlarge's AMD EPYC 7R13 Processor
(12 cores, 24 threads). llama.cpp 2f729cf (release b11672, MIT),
built on the instance, the official checkpoint converted to GGUF BF16 by its
own script, `llama-bench`, 128 tokens generated per test, 5 repetitions,
12 threads. `transformers` 5.18.0: BF16, sdpa
attention, 128 tokens, median of 5.

**How to read this:** the columns are two separate ways of running the
same model, not one on top of the other.

- llama.cpp runs SmolLM2 on the host's CPU alone, without the FPGA.
- The core runs it on the FPGA; the CPU only hands it a token and reads
  back the result each step.

A negative share means the core is that much slower than the CPU running
the model by itself. Using the core does not slow the CPU down.

This first build does not yet beat a 12-core server CPU on a model this
small. The reasons (see "What limits it now" above):

- **Clock:** it runs at 100 MHz, not the 250 MHz it was designed for (M10).
- **Vector unit:** its work comes after the memory reads instead of beside
  them, which keeps the core at about half of what the memory bandwidth
  allows.
- **Logits' read:** in wall-clock time, reading the logits back over PCIe
  takes longer than the core's own work (Q-34).

| core positions | Cabosse core, 100 MHz | llama.cpp positions | llama.cpp on the CPU | core vs llama.cpp alone |
|---|---|---|---|---|
| 0–127 | 194.8 | 0–127 | 244.1 ± 5.3 | -20% |
| 46–173 | 192.7 | 46–173 | 221.9 ± 3.5 | -13% |
| 917–1044 | 159.0 | 1023–1150 | 169.5 ± 9.6 | -6% |

The core's figures come from its cycle counter on the long run, per
position; llama.cpp's are `llama-bench`'s mean ± standard deviation. The
core's last window ends at 1044, the end of the long run, slightly before
llama.cpp's (which favours the core a little). `transformers` on the same
CPU: 75.8 tokens/s (positions 5–132, median of 5, started right after
`llama-bench`, at a 1-minute load average of 7.5).

At 100 MHz the core is 6–20% behind a well-optimized CPU library on 12
server cores. The rows differ in more than speed: the core is bit-exact
against a stated reference with FP32 accumulation; llama.cpp keeps its KV
cache in FP16 (its default) and has its own summation order. The design's
target is 250 MHz (the next milestone).

Logs: [baseline/](../reports/data/f2/2026-11-05-core-run3/baseline/).

### History

| build | date | clock | tokens/s, core (64 tokens) | tokens/s, end to end | bit-exact |
|---|---|---|---|---|---|
| core build 1 | 2026-11-05 | 100 MHz | 194.5 | 68.0 | yes |

## What it is built with, and the licences

Everything Cabosse wrote is open source. **The closed parts are AMD's
Vivado, the AMD IP cores it generates, and the AWS F2 shell.** No FPGA
in AWS can be built or loaded without them; an open flow would need a
different FPGA (out of the project's scope for now).

| component | used for | licence | open source |
|---|---|---|---|
| Cabosse RTL (`rtl/`, `platforms/`) | the accelerator | Solderpad Hardware 2.1 (Apache-2.0 based) | yes |
| Cabosse software, golden model, tests | runtime, reference, verification | Apache-2.0 | yes |
| SmolLM2-135M-Instruct weights | the model | Apache-2.0 | yes (open weights) |
| PyTorch | golden model | BSD-3-Clause | yes |
| Hugging Face `transformers`, `tokenizers` | reference model, tokenizer | Apache-2.0 | yes |
| Verilator | RTL simulation and lint | LGPL-3.0 or Artistic-2.0 | yes |
| cocotb | RTL testbenches | BSD-3-Clause | yes |
| Yosys | synthesis checks | ISC | yes |
| aws-fpga HDK/SDK | F2 build scripts, host libraries | Apache-2.0 | yes |
| llama.cpp (b11672) | CPU baseline, not part of Cabosse | MIT | yes |
| uv, Python | Python environment | Apache-2.0 or MIT; PSF-2.0 | yes |
| **AMD Vivado 2025.2** | FPGA synthesis, place and route | proprietary (licence included with AWS's FPGA Developer AMI) | **no** |
| **AMD IP cores** (HBM, AXI SmartConnect, clocking, debug) | inside the F2 design, generated by Vivado | proprietary, under the Vivado licence | **no** |
| **AWS F2 shell** | PCIe and HBM interface around the design | provided by AWS as a compiled checkpoint | **no** |

## Reproduce

- Build: [scripts/f2/build_cl.sh](../scripts/f2/build_cl.sh) with `DESIGN=core`, on a c6i.8xlarge.
- Run: [scripts/f2/run_core.sh](../scripts/f2/run_core.sh) on an f2.6xlarge with the AFI loaded.
- The chat on a CPU, no FPGA: `python scripts/f2/chat.py OUT --golden`.
- The animation: `python scripts/cast_to_svg.py chat.cast chat-f2.svg`.
