"""建立 V2 Git 證據與自包含審查 ZIP。"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from prospective.canonical import read_ledger, sha256_file
from .config import DATA_PATH, LEDGER_PATH, ROOT, SOURCE_FILES, STATE_DIR, TIMEZONE
from .remote import branch_name, git, ls_remote_oid, remote_repository

ZIP_PATH = STATE_DIR / "prospective_v2_setup_review.zip"
GIT_BUNDLE_PATH = STATE_DIR / "prospective-v2-history.bundle"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=check)


def prepare_git_evidence() -> dict:
    records = read_ledger(LEDGER_PATH)
    predictions = [r for r in records if r.get("event_type") == "prediction"]
    anchors = [r for r in records if r.get("event_type") == "remote_anchor"]
    if not predictions or not anchors: raise RuntimeError("缺少 prediction 或 remote_anchor")
    prediction = predictions[0]; anchor = anchors[0]
    if GIT_BUNDLE_PATH.exists(): GIT_BUNDLE_PATH.unlink()
    git(ROOT, "bundle", "create", str(GIT_BUNDLE_PATH), "--all")
    verify = git(ROOT, "bundle", "verify", str(GIT_BUNDLE_PATH), check=False)
    (STATE_DIR / "git_bundle_verify.txt").write_text(verify.stdout + verify.stderr, encoding="utf-8")
    if verify.returncode != 0: raise RuntimeError("git bundle verify 失敗")
    log = git(ROOT, "log", "--all", "--decorate", "--graph", "--format=fuller").stdout
    (STATE_DIR / "git_log_full.txt").write_text(log, encoding="utf-8")
    (STATE_DIR / "git_remote.txt").write_text(git(ROOT, "remote", "-v").stdout, encoding="utf-8")
    refs = git(ROOT, "ls-remote", "origin").stdout
    tags = git(ROOT, "ls-remote", "--tags", "origin").stdout
    (STATE_DIR / "git_remote_refs.txt").write_text(refs, encoding="utf-8")
    (STATE_DIR / "git_remote_tags.txt").write_text(tags, encoding="utf-8")
    (STATE_DIR / "prediction_commit.txt").write_text(git(ROOT, "show", "--stat", "--format=fuller", anchor["prediction_commit"]).stdout, encoding="utf-8")
    anchor_commit = git(ROOT, "log", "-1", "--format=%H", "--", str(LEDGER_PATH.relative_to(ROOT))).stdout.strip()
    (STATE_DIR / "remote_anchor_commit.txt").write_text(git(ROOT, "show", "--stat", "--format=fuller", anchor_commit).stdout, encoding="utf-8")
    repository = remote_repository(); repo_slug = repository.removesuffix(".git").split("github.com/")[-1]
    api = _run(["gh", "api", f"repos/{repo_slug}/commits/{anchor['prediction_commit']}"])
    api_payload = json.loads(api.stdout)
    write_time = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
    (STATE_DIR / "remote_verification_raw.json").write_text(json.dumps(api_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    branch = branch_name(); current_oid = ls_remote_oid(ROOT, "origin", f"refs/heads/{branch}")
    verification = {"queried_at": write_time, "remote_repository": repository, "remote_branch": branch,
                    "current_remote_ref_oid": current_oid, "prediction_commit": anchor["prediction_commit"],
                    "prediction_remote_ref_oid_at_anchor": anchor["remote_ref_oid"],
                    "anchor_commit": anchor_commit, "github_api_commit_sha": api_payload.get("sha"),
                    "github_commit_url": anchor["remote_commit_url"], "verification_method": ["git-ls-remote", "GitHub REST API"],
                    "prediction_commit_api_verified": api_payload.get("sha") == anchor["prediction_commit"],
                    "git_bundle_verify_passed": True}
    (STATE_DIR / "remote_verification.json").write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return verification


def verify_zip(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist(); manifest = json.loads(archive.read("file_manifest.json"))
        errors = []
        if len(names) != len(set(names)): errors.append("duplicate entries")
        for item in manifest:
            try: content = archive.read(item["relative_path"])
            except KeyError: errors.append("missing:" + item["relative_path"]); continue
            if len(content) != item["bytes"]: errors.append("size:" + item["relative_path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]: errors.append("sha256:" + item["relative_path"])
        expected = {item["relative_path"] for item in manifest} | {"file_manifest.json", "file_manifest.csv", "zip_validation.json"}
        if set(names) != expected: errors.append("entry set differs from manifest metadata")
    return {"validation_passed": not errors, "errors": errors, "manifest_payload_count": len(manifest),
            "zip_entry_count": len(names)}


def export_review_bundle() -> dict:
    prepare_git_evidence()
    with tempfile.TemporaryDirectory(prefix="v2-review-") as temporary:
        stage = Path(temporary)
        copy_paths = [*SOURCE_FILES, "prospective_validation_v1_closure.md", "prospective_validation_v1_closure.json",
                      "tests/test_prospective_v2.py"]
        for relative in copy_paths:
            source = ROOT / relative; destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        for source in sorted(STATE_DIR.rglob("*")):
            if not source.is_file() or source == ZIP_PATH or source.name.startswith("full-reproduce-"): continue
            destination = stage / source.relative_to(ROOT); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
        data_destination = stage / "data" / "processed" / "lotto649.csv"; data_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA_PATH, data_destination)
        freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"], check=False)
        (stage / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
        (stage / "environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\n", encoding="utf-8")
        (stage / "README_V2.md").write_text("# V2 remote time anchor review bundle\n\n模型方法與V1相同；正式性只由合法remote_anchor判定。\n", encoding="utf-8")
        (stage / "reproduce_prediction.py").write_text(
            "import json\nfrom next_draw_predictor_audited import load_lotto_data\nfrom prospective.model import fit_predict\n"
            "from prospective_v2.workflow import compare_reproduction\n"
            "from pathlib import Path\ndata=load_lotto_data('data/processed/lotto649.csv')\n"
            "path=next(Path('prospective_validation_v2/predictions').glob('prediction-*.json'))\n"
            "expected=json.loads(path.read_text(encoding='utf-8')); actual=fit_predict(data)\n"
            "compare_reproduction(expected,actual,atol=1e-12)\nprint('V2_PREDICTION_REPRODUCED')\n", encoding="utf-8")
        (stage / "verify_manifest.py").write_text(
            "import hashlib,json\nfrom pathlib import Path\nr=Path('.')\n"
            "m=json.loads((r/'file_manifest.json').read_text(encoding='utf-8'))\n"
            "assert all((r/i['relative_path']).is_file() and (r/i['relative_path']).stat().st_size==i['bytes'] and hashlib.sha256((r/i['relative_path']).read_bytes()).hexdigest()==i['sha256'] for i in m)\n"
            "print(f'MANIFEST_VERIFIED: {len(m)}')\n", encoding="utf-8")
        (stage / "reproduce.ps1").write_text(
            "$ErrorActionPreference='Stop'\nif(Get-Command py -ErrorAction SilentlyContinue){py -3 -m venv .venv}else{python -m venv .venv}\n"
            "if(Test-Path '.\\.venv\\Scripts\\python.exe'){$p='.\\.venv\\Scripts\\python.exe'}elseif(Test-Path '.\\.venv\\bin\\python'){$p='.\\.venv\\bin\\python'}else{throw 'venv python missing'}\n"
            "& $p -m pip install -r requirements-lock.txt\n& $p -m pytest -q tests/test_prospective_v2.py --basetemp .pytest-tmp\n"
            "& $p -m prospective_v2.verify_ledger\n& $p reproduce_prediction.py\n& git bundle verify prospective_validation_v2/prospective-v2-history.bundle\n& $p verify_manifest.py\n", encoding="utf-8")
        excluded = {"file_manifest.json", "file_manifest.csv", "zip_validation.json"}
        files = [p for p in sorted(stage.rglob("*")) if p.is_file() and p.name not in excluded]
        manifest = [{"relative_path": p.relative_to(stage).as_posix(), "bytes": p.stat().st_size,
                     "sha256": sha256_file(p), "description": "V2 review evidence"} for p in files]
        (stage / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with (stage / "file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["relative_path","bytes","sha256","description"]); writer.writeheader(); writer.writerows(manifest)
        expected_entries = len(manifest) + 3
        validation = {"validation_passed": True, "manifest_payload_count": len(manifest), "zip_entry_count": expected_entries}
        (stage / "zip_validation.json").write_text(json.dumps(validation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(ZIP_PATH,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for p in sorted(stage.rglob("*")):
                if p.is_file(): archive.write(p,p.relative_to(stage).as_posix())
    result = verify_zip(ZIP_PATH)
    return {"path": str(ZIP_PATH.resolve()), "bytes": ZIP_PATH.stat().st_size,
            "sha256": sha256_file(ZIP_PATH), **result}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--verify-only")
    args=parser.parse_args(); print(json.dumps(verify_zip(args.verify_only) if args.verify_only else export_review_bundle(),ensure_ascii=False,indent=2))
