"""Compatibility re-export shim for `app.rag.legal_v2.ingest.source_inventory`.

Implementation lives in `app.rag.legal_v2.ingest.source_inventory`.
"""

from __future__ import annotations

from importlib import import_module
import sys

_impl = import_module("app.rag.legal_v2.ingest.source_inventory")
sys.modules[__name__] = _impl
