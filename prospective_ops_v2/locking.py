"""Auditable exclusive-operation locks with conservative stale recovery."""
from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .verify import EvidenceError


def _process_is_alive(pid: int) -> bool | None:
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if process:
            ctypes.windll.kernel32.CloseHandle(process)
            return True
        error = ctypes.windll.kernel32.GetLastError()
        if error == 87:  # ERROR_INVALID_PARAMETER: no process owns this PID.
            return False
        if error == 5:  # ERROR_ACCESS_DENIED: the process exists but is protected.
            return True
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def inspect_lock(path: str | Path) -> dict[str, object]:
    lock = Path(path)
    if not lock.exists():
        return {"exists": False, "recoverable": False}
    try:
        owner = json.loads(lock.read_text(encoding="utf-8"))
        pid = int(owner["pid"])
        hostname = str(owner["hostname"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"exists": True, "recoverable": False, "reason": "owner metadata is not trustworthy"}
    same_host = hostname == socket.gethostname()
    alive = _process_is_alive(pid) if same_host else None
    return {
        "exists": True,
        "recoverable": same_host and alive is False,
        "owner_pid": pid,
        "owner_hostname": hostname,
        "owner_alive": alive,
        "reason": "owner process is provably absent" if same_host and alive is False else "owner may still be active",
    }


def recover_stale_lock(path: str | Path) -> dict[str, object]:
    lock = Path(path)
    status = inspect_lock(lock)
    if not status.get("recoverable"):
        raise EvidenceError(f"refusing stale-lock recovery: {status.get('reason', 'not recoverable')}")
    lock.unlink()
    return {**status, "removed": True, "path": str(lock)}


@contextmanager
def exclusive_operation_lock(path: str | Path, *, purpose: str) -> Iterator[None]:
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        status = inspect_lock(lock)
        guidance = "python -m prospective_ops_v2.complete_cycle --draw-id <DRAW_ID> --recover-stale-lock"
        raise EvidenceError(
            f"another operation lock exists ({status.get('reason')}); manual recovery only: {guidance}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
        if current == metadata and lock.exists():
            lock.unlink()
