"""實際 Git remote ref 驗證與 ledger anchor 解析。"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import ROOT


def git(root: str | Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=check)


def branch_name(root: str | Path = ROOT) -> str:
    return git(root, "branch", "--show-current").stdout.strip()


def remote_repository(root: str | Path = ROOT, remote: str = "origin") -> str:
    return git(root, "remote", "get-url", remote).stdout.strip()


def ls_remote_oid(root: str | Path, remote: str, ref: str) -> str | None:
    result = git(root, "ls-remote", remote, ref, check=False)
    if result.returncode != 0:
        return None
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    matches = [oid for oid, found_ref in rows if found_ref == ref]
    return matches[0] if len(matches) == 1 else None


def require_remote_oid(root: str | Path, remote: str, ref: str, expected_oid: str) -> str:
    actual = ls_remote_oid(root, remote, ref)
    if actual != expected_oid:
        raise RuntimeError(f"遠端 ref OID 不符：expected={expected_oid}, actual={actual}")
    return actual


def commit_url(repository: str, commit: str) -> str:
    clean = repository.removesuffix(".git")
    if clean.startswith("git@github.com:"):
        clean = "https://github.com/" + clean.removeprefix("git@github.com:")
    if not clean.startswith("https://github.com/"):
        raise ValueError("remote repository 不是可公開驗證的 GitHub URL")
    return f"{clean}/commit/{commit}"


def _is_remote_descendant(root: str | Path, remote: str, branch: str,
                          prediction_commit: str, current_remote_oid: str) -> bool:
    fetch = git(root, "fetch", "--quiet", remote, f"refs/heads/{branch}", check=False)
    if fetch.returncode != 0:
        return False
    return git(root, "merge-base", "--is-ancestor", prediction_commit, current_remote_oid, check=False).returncode == 0


def resolve_remote_anchor(prediction_id: str, records: list[dict[str, Any]], *,
                          root: str | Path = ROOT, revalidate_remote: bool = True) -> dict[str, Any] | None:
    predictions = [record for record in records if record.get("event_type") == "prediction"
                   and record.get("prediction_id") == prediction_id]
    if len(predictions) != 1:
        return None
    prediction = predictions[0]
    anchors = [record for record in records if record.get("event_type") == "remote_anchor"
               and record.get("prediction_id") == prediction_id]
    for anchor in reversed(anchors):
        if anchor.get("prediction_record_hash") != prediction.get("record_hash"):
            continue
        if not anchor.get("verified_before_draw"):
            continue
        if anchor.get("official_draw_status_at_verification") != "NOT_ANNOUNCED":
            continue
        if anchor.get("remote_ref_oid") != anchor.get("prediction_commit"):
            continue
        if revalidate_remote:
            branch = str(anchor.get("remote_branch"))
            ref = f"refs/heads/{branch}"
            current = ls_remote_oid(root, str(anchor.get("remote_name", "origin")), ref)
            if not current or not _is_remote_descendant(root, str(anchor.get("remote_name", "origin")),
                                                        branch, str(anchor["prediction_commit"]), current):
                continue
        return anchor
    return None
