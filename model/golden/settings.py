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
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    ftz: bool = False  # flush subnormal inputs and outputs to signed zero (D-016: off)


_DEFAULTS = Settings()
_current: ContextVar[Settings] = ContextVar("settings")


def current() -> Settings:
    return _current.get(_DEFAULTS)


@contextmanager
def override(**changes: object) -> Generator[None]:
    """Change settings inside a `with` block; the previous values come back after it."""
    token = _current.set(replace(current(), **changes))  # unknown names raise TypeError
    try:
        yield
    finally:
        _current.reset(token)
