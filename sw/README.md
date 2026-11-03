# sw/

Software that runs on the host CPU and talks to the accelerator.

- `runtime.py`: lays a model out in the accelerator's memory
  (`commands.Layout`), validates and writes its command list once, then
  runs one decode step per token with the token id and the position in
  registers. Greedy generation; at the end of the context it stops the
  conversation with a message (PLAN.md, Non-goals).
- `platforms.py`: the six calls a platform gives the runtime (registers,
  memory, run until done, read the logits).
- `golden_platform.py`: a platform on the golden model's reference
  controller (`commands.run()`), to test the runtime without RTL.

Planned: the simulation platform (a Verilator driver in a child process,
M7) and F2's (M9), on the same six calls; a PyTorch integration later.
