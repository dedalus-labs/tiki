# Copyright © 2026 Dedalus Labs, Inc.

"""Tiki: an accelerated array framework. The compiled core is the top-level API."""

import sys as _sys
from types import ModuleType as _ModuleType

from tiki import core as _core
from tiki.core import *  # noqa: F401,F403
from tiki.core import __array_namespace_info__, __version__  # noqa: F401

# The core's submodules are the public ones: ``import tiki.random`` names the
# same module object as ``tiki.core.random``.
for _name, _member in vars(_core).items():
    if isinstance(_member, _ModuleType) and not _name.startswith("_"):
        _sys.modules[f"tiki.{_name}"] = _member
