"""Compatibility re-export shim for ``app.rag.legal_v2.interpret``.

Implementation lives in ``app.rag.legal_v2.interpret.interpreter``.
"""

from __future__ import annotations

from importlib import import_module
import sys

_impl = import_module("app.rag.legal_v2.interpret.interpreter")
sys.modules[__name__] = _impl
