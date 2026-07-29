"""Two-phase remote anchoring for result and correction events."""
from __future__ import annotations

import uuid
import subprocess
import json
from pathlib import Path
from typing import Any

from prospective_v2.config import EXPERIMENT_ID, EXPERIMENT_VERSION
from prospective_v2.remote import commit_url, ls_remote_oid
from prospective.canonical import ZERO_HASH, canonical_bytes, hash_record

from .result_events import now_iso
from .verify import EvidenceError


def build_result_remote_anchor(
    result: dict[str, Any],
    result_commit: str,
    *,
    root: str | Path,
    remote: str,
    branch: str,
    repository: str,
    ledger_relative_path: str = "prospective_validation_v2/ledger.jsonl",
    head_relative_path: str = "prospective_validation_v2/ledger_head.json",
    verified_at: str | None = None,
) -> dict[str, Any]:
    ref = f"refs/heads/{branch}"
    remote_oid = ls_remote_oid(root, remote, ref)
    if not remote_oid:
        raise EvidenceError("result commit has not been pushed")
    if remote_oid != result_commit:
        raise EvidenceError(f"result remote OID mismatch: expected={result_commit}, actual={remote_oid}")
    configured_repository = subprocess.run(
        ["git", "remote", "get-url", remote], cwd=root, text=True, capture_output=True,
    ).stdout.strip()
    if configured_repository.removesuffix(".git") != repository.removesuffix(".git"):
        raise EvidenceError("anchor repository does not match the configured Git remote")
    fetch = subprocess.run(
        ["git", "fetch", "--quiet", remote, ref], cwd=root, capture_output=True,
    )
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{result_commit}^{{commit}}"], cwd=root, capture_output=True,
    )
    if fetch.returncode != 0 or exists.returncode != 0:
        raise EvidenceError("pushed result commit cannot be fetched and resolved")
    _verify_result_commit(root, result_commit, result, ledger_relative_path, head_relative_path)
    if repository.startswith(("https://github.com/", "git@github.com:")):
        url = commit_url(repository, result_commit)
    else:
        url = Path(repository).resolve().as_uri() + f"?commit={result_commit}"
    return {
        "event_type": "result_remote_anchor",
        "event_id": str(uuid.uuid4()),
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "target_draw_id": str(result["target_draw_id"]),
        "result_event_type": result["event_type"],
        "result_record_hash": result["record_hash"],
        "result_commit": result_commit,
        "remote_name": remote,
        "remote_repository": repository,
        "remote_branch": branch,
        "remote_ref_oid": remote_oid,
        "remote_commit_url": url,
        "remote_verified_at": verified_at or now_iso(),
        "verification_method": "git-ls-remote",
        "result_ledger_path": ledger_relative_path,
        "result_ledger_head_path": head_relative_path,
    }


def resolve_result_anchor(
    result: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    root: str | Path,
    revalidate_remote: bool = True,
    expected_repository: str | None = None,
    expected_branch: str | None = None,
    ledger_relative_path: str = "prospective_validation_v2/ledger.jsonl",
    head_relative_path: str = "prospective_validation_v2/ledger_head.json",
) -> dict[str, Any] | None:
    anchors = [
        record for record in records
        if record.get("event_type") == "result_remote_anchor"
        and record.get("result_record_hash") == result.get("record_hash")
        and str(record.get("target_draw_id")) == str(result.get("target_draw_id"))
    ]
    for anchor in reversed(anchors):
        required = ("remote_repository", "remote_branch", "remote_ref_oid", "remote_commit_url",
                    "remote_verified_at", "result_ledger_path", "result_ledger_head_path")
        if any(not anchor.get(field) for field in required):
            continue
        if anchor.get("result_ledger_path") != ledger_relative_path or anchor.get("result_ledger_head_path") != head_relative_path:
            continue
        if expected_repository is not None and str(anchor.get("remote_repository")).removesuffix(".git") != expected_repository.removesuffix(".git"):
            continue
        if expected_branch is not None and anchor.get("remote_branch") != expected_branch:
            continue
        if anchor.get("result_commit") != anchor.get("remote_ref_oid"):
            continue
        if anchor.get("verification_method") != "git-ls-remote":
            continue
        try:
            _verify_result_commit(root, str(anchor["result_commit"]), result,
                                  ledger_relative_path, head_relative_path)
        except EvidenceError:
            continue
        if revalidate_remote:
            branch = str(anchor.get("remote_branch"))
            current = ls_remote_oid(root, str(anchor.get("remote_name", "origin")), f"refs/heads/{branch}")
            if not current:
                continue
            # The branch may advance after anchoring. The anchored commit must remain an ancestor.
            fetch = subprocess.run(
                ["git", "fetch", "--quiet", str(anchor.get("remote_name", "origin")), f"refs/heads/{branch}"],
                cwd=root, capture_output=True,
            )
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", str(anchor["result_commit"]), current],
                cwd=root, capture_output=True,
            )
            if fetch.returncode != 0 or ancestor.returncode != 0:
                continue
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{anchor['result_commit']}^{{commit}}"], cwd=root, capture_output=True,
            )
            if exists.returncode != 0:
                continue
        return anchor
    return None


def _git_show(root: str | Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"], cwd=root, capture_output=True,
    )
    if result.returncode != 0:
        raise EvidenceError(f"anchored commit is missing {relative_path}")
    return result.stdout


def _verify_result_commit(root: str | Path, commit: str, expected_result: dict[str, Any],
                          ledger_relative_path: str, head_relative_path: str) -> None:
    """Verify the exact canonical ledger and head stored by the anchored commit."""
    ledger_bytes = _git_show(root, commit, ledger_relative_path)
    head_bytes = _git_show(root, commit, head_relative_path)
    previous = ZERO_HASH; matches: list[dict[str, Any]] = []; count = 0
    for count, raw_line in enumerate(ledger_bytes.splitlines(), 1):
        try:
            line = raw_line.decode("utf-8"); record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError("anchored ledger is not valid canonical JSONL") from exc
        if canonical_bytes(record) != raw_line:
            raise EvidenceError("anchored ledger contains non-canonical JSON")
        if record.get("sequence") != count or record.get("previous_record_hash") != previous:
            raise EvidenceError("anchored ledger chain sequence/previous hash is invalid")
        if record.get("record_hash") != hash_record(record):
            raise EvidenceError("anchored ledger record hash is invalid")
        previous = record["record_hash"]
        if record.get("record_hash") == expected_result.get("record_hash"):
            matches.append(record)
    try:
        head = json.loads(head_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("anchored ledger head is invalid") from exc
    if head != {"record_count": count, "record_hash": previous}:
        raise EvidenceError("anchored ledger head does not match the committed ledger")
    if len(matches) != 1 or matches[0] != expected_result:
        raise EvidenceError("anchored commit does not contain the exact result record")
