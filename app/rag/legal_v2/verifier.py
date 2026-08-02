"""Compatibility re-export shim for ``app.rag.legal_v2.verify``.

Implementation lives in ``app.rag.legal_v2.verify.verifier``. This module keeps
legacy import paths working for the API, pipeline, scripts, and tests.
"""

from __future__ import annotations

from importlib import import_module
import sys

_impl = import_module("app.rag.legal_v2.verify.verifier")
sys.modules[__name__] = _impl
