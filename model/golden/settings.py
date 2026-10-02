# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Cabosse Authors
"""Golden-model settings that are not part of the numerics spec.

They exist for experiments, e.g. measuring what flush-to-zero would change.
The defaults are the spec. Change settings only temporarily:

    from golden import settings
    with settings.override(ftz=True):
        ...
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Settings:
    ftz: bool = False  # flush subnormal inputs and outputs to signed zero (D-016: off)


current = Settings()


@contextmanager
def override(**changes: object) -> Generator[None]:
    """Change settings inside a `with` block; the previous values come back after it."""
    unknown = set(changes) - set(vars(current))
    if unknown:
        raise AttributeError(f"unknown settings: {sorted(unknown)}")
    old = {name: getattr(current, name) for name in changes}
    for name, value in changes.items():
        setattr(current, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(current, name, value)
