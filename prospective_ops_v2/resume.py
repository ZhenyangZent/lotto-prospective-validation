"""Persistent-state orchestration primitives for retry-safe V2 operations."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class CycleState(str, Enum):
    NO_RESULT = "NO_RESULT"
    RESULT_APPENDED_LOCAL = "RESULT_APPENDED_LOCAL"
    RESULT_COMMIT_LOCAL_ONLY = "RESULT_COMMIT_LOCAL_ONLY"
    RESULT_COMMIT_REMOTE = "RESULT_COMMIT_REMOTE"
    RESULT_ANCHORED_LOCAL = "RESULT_ANCHORED_LOCAL"
    RESULT_ANCHOR_REMOTE = "RESULT_ANCHOR_REMOTE"
    CSV_UPDATED_LOCAL = "CSV_UPDATED_LOCAL"
    CSV_COMMIT_LOCAL_ONLY = "CSV_COMMIT_LOCAL_ONLY"
    CSV_COMMIT_REMOTE = "CSV_COMMIT_REMOTE"
    NEXT_PREDICTION_LOCAL = "NEXT_PREDICTION_LOCAL"
    NEXT_PREDICTION_REMOTE = "NEXT_PREDICTION_REMOTE"
    NEXT_PREDICTION_ANCHORED = "NEXT_PREDICTION_ANCHORED"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"


STATE_ORDER = tuple(CycleState)


@dataclass(frozen=True)
class CycleObservation:
    state: CycleState
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumableCycleServices:
    inspect: Callable[[str], CycleObservation]
    actions: Mapping[CycleState, Callable[[str, CycleObservation], Any]]
    preflight: Callable[[str], Any] = lambda _draw_id: None
    postflight: Callable[[str], Any] = lambda _draw_id: None


def execute_resumable_cycle(
    draw_id: str,
    services: ResumableCycleServices,
    *,
    max_transitions: int = 32,
) -> dict[str, Any]:
    """Resume at the first incomplete durable state and require observable progress."""
    requested = str(draw_id)
    services.preflight(requested)
    transitions: list[dict[str, Any]] = []
    for _ in range(max_transitions + 1):
        before = services.inspect(requested)
        if before.state is CycleState.CYCLE_COMPLETE:
            services.postflight(requested)
            return {
                "draw_id": requested,
                "state": before.state.value,
                "idempotent": len(transitions) == 0,
                "transitions": transitions,
                "details": dict(before.details),
            }
        action = services.actions.get(before.state)
        if action is None:
            raise RuntimeError(f"no safe resume action is registered for {before.state.value}")
        value = action(requested, before)
        after = services.inspect(requested)
        if STATE_ORDER.index(after.state) <= STATE_ORDER.index(before.state):
            raise RuntimeError(
                f"resume action made no durable forward progress: {before.state.value} -> {after.state.value}"
            )
        transitions.append({
            "from": before.state.value,
            "to": after.state.value,
            "result": value,
        })
    raise RuntimeError(f"cycle exceeded {max_transitions} durable transitions")


def git_output(root: str | Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=Path(root), text=True, capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def is_ancestor(root: str | Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=Path(root), capture_output=True,
    ).returncode == 0


def require_fast_forward_push_topology(
    root: str | Path,
    *,
    remote: str,
    branch: str,
    local_oid: str,
    remote_oid: str,
) -> None:
    """Allow equality or a normal fast-forward push; reject remote-ahead/divergence."""
    if local_oid == remote_oid:
        return
    if is_ancestor(root, remote_oid, local_oid):
        return
    if is_ancestor(root, local_oid, remote_oid):
        raise RuntimeError(
            "remote branch contains an unknown newer commit; inspect it and fast-forward manually"
        )
    raise RuntimeError("local HEAD and remote branch have diverged; force push is prohibited")


def find_commit_with_exact_bytes(
    root: str | Path,
    relative_path: str,
    required_bytes: bytes,
) -> str | None:
    """Return the newest local commit whose file contains the exact required bytes."""
    commits = git_output(root, "rev-list", "HEAD", "--", relative_path).splitlines()
    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"], cwd=Path(root), capture_output=True,
        )
        if result.returncode == 0 and required_bytes in result.stdout:
            return commit
    return None


def find_commit_with_exact_file(
    root: str | Path,
    relative_path: str,
    expected_bytes: bytes,
) -> str | None:
    """Return the newest local commit with exactly the expected file bytes."""
    commits = git_output(root, "rev-list", "HEAD", "--", relative_path).splitlines()
    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"], cwd=Path(root), capture_output=True,
        )
        if result.returncode == 0 and result.stdout == expected_bytes:
            return commit
    return None


def commit_is_remote(root: str | Path, commit: str, remote_oid: str) -> bool:
    return bool(commit and remote_oid and is_ancestor(root, commit, remote_oid))
