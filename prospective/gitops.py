"""窄範圍 Git 防護與提交操作。"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_git(root: str | Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=check)


def is_repository(root: str | Path) -> bool:
    return run_git(root, "rev-parse", "--is-inside-work-tree", check=False).returncode == 0


def current_commit(root: str | Path) -> str:
    result = run_git(root, "rev-parse", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else "GIT_COMMIT_UNAVAILABLE"


def assert_clean(root: str | Path) -> None:
    if not is_repository(root):
        raise RuntimeError("Git repository 不存在")
    output = run_git(root, "status", "--porcelain").stdout.strip()
    if output:
        raise RuntimeError(f"Git working tree 不乾淨：\n{output}")


def remote_url(root: str | Path) -> str | None:
    result = run_git(root, "remote", "get-url", "origin", check=False)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def remote_commit_url(root: str | Path, commit: str) -> str:
    remote = remote_url(root)
    if not remote:
        return "REMOTE_COMMIT_NOT_CONFIRMED"
    clean = remote.removesuffix(".git")
    if clean.startswith("git@github.com:"):
        clean = "https://github.com/" + clean.removeprefix("git@github.com:")
    return f"{clean}/commit/{commit}" if clean.startswith("http") else "REMOTE_COMMIT_NOT_CONFIRMED"


def commit_paths(root: str | Path, paths: list[str], message: str) -> str:
    run_git(root, "add", "--", *paths)
    run_git(root, "commit", "-m", message)
    return current_commit(root)


def push(root: str | Path, *, tags: bool = False) -> bool:
    if not remote_url(root):
        return False
    args = ["push", "origin", "HEAD"]
    if tags:
        args.append("--follow-tags")
    return run_git(root, *args, check=False).returncode == 0
