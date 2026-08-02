"""Compatibility re-export shim for `app.rag.legal_v2.ingest.chunking`.

Implementation lives in `app.rag.legal_v2.ingest.chunking`.
"""

from __future__ import annotations

from importlib import import_module
import sys

_impl = import_module("app.rag.legal_v2.ingest.chunking")
sys.modules[__name__] = _impl
