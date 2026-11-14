# docs/

Design documents that the code is checked against.

- `numerics.md`: the numerics spec, i.e. the contract: formats, rounding,
  summation order, special values, function approximations (M1). The golden
  model and the RTL both implement it.
- `architecture.md`: block diagram, interfaces, memory map, command format,
  and data layouts (M2).
- `perf.md`: first-order performance model and its assumptions (M2).
- `f2.md`: facts about the AWS F2 platform, with sources (M2).
- `f2-chat.md`: how to chat with the model on an F2 instance (M11).
- `aws-setup.md`: one-time AWS account setup for F2 work.

The project plan and decision log live in [`../PLAN.md`](../PLAN.md).
