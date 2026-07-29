"""Append-only operations for the frozen V2 prospective experiment.

This package is intentionally outside :mod:`prospective_v2.config.SOURCE_FILES`.
It operates on ledger copies in tests/dry-runs and never changes the frozen model.
"""

from .result_events import build_result_event, ingest_result
from .result_remote_anchor import build_result_remote_anchor, resolve_result_anchor

__all__ = [
    "build_result_event",
    "ingest_result",
    "build_result_remote_anchor",
    "resolve_result_anchor",
]
