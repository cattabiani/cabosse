# sw/

Software that runs on the host CPU and talks to the accelerator.

Planned contents:

- **Runtime:** loads a checkpoint, lays out weights in accelerator memory,
  builds the command stream, runs decode, and reads back results.
- **Driver/platform glue:** register access and DMA for each platform
  (simulation, AWS F2).
- **PyTorch integration:** a later, optional way to call the accelerator from
  PyTorch.

The same runtime should drive simulation and real hardware, so tests written
against simulation can be rerun on the FPGA.
