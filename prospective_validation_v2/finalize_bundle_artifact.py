"""修整 V2 review ZIP 的自包含依賴與 fail-fast 重現腳本。

此檔是封裝產物工具，不屬於已凍結 prediction/anchor source hash。
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).with_name("prospective_v2_setup_review.zip")

REPRODUCE = r'''$ErrorActionPreference='Stop'
$pyCommand=Get-Command py -ErrorAction SilentlyContinue
if($pyCommand){$launcher=$pyCommand.Source;$launcherArgs=@('-3')}else{$launcher=(Get-Command python -ErrorAction Stop).Source;$launcherArgs=@()}
& $launcher @launcherArgs -m venv .venv
$venvPython=$null
if(Test-Path '.\.venv\Scripts\python.exe'){$venvPython='.\.venv\Scripts\python.exe'}elseif(Test-Path '.\.venv\bin\python.exe'){$venvPython='.\.venv\bin\python.exe'}elseif(Test-Path '.\.venv\bin\python'){$venvPython='.\.venv\bin\python'}
$venvHealthy=$false
if($venvPython){& $venvPython -c 'import sys; print(sys.executable)' *> $null;$venvHealthy=($LASTEXITCODE -eq 0)}
if($venvHealthy){$p=$venvPython;$pArgs=@(); & $p -m pip install -r requirements-lock.txt}else{$p=$launcher;$pArgs=$launcherArgs;New-Item -ItemType Directory -Force -Path '.repro-packages'|Out-Null;& $p @pArgs -m pip install -r requirements-lock.txt --target .repro-packages;$env:PYTHONPATH=(Resolve-Path '.repro-packages').Path}
if($LASTEXITCODE -ne 0){throw 'dependency installation failed'}
$pytestTemp=Join-Path ([IO.Path]::GetTempPath()) ('prospective-v2-pytest-'+[guid]::NewGuid().ToString('N'))
try{& $p @pArgs -m pytest -q tests/test_prospective_v2.py --basetemp $pytestTemp;$testExit=$LASTEXITCODE}finally{if(Test-Path -LiteralPath $pytestTemp){Remove-Item -LiteralPath $pytestTemp -Recurse -Force}}
if($testExit -ne 0){throw 'pytest failed'}
& $p @pArgs -m prospective_v2.verify_ledger
if($LASTEXITCODE -ne 0){throw 'ledger verification failed'}
& $p @pArgs reproduce_prediction.py
if($LASTEXITCODE -ne 0){throw 'prediction reproduction failed'}
git bundle verify prospective_validation_v2/prospective-v2-history.bundle
if($LASTEXITCODE -ne 0){throw 'git bundle verification failed'}
& $p @pArgs verify_manifest.py
if($LASTEXITCODE -ne 0){throw 'manifest verification failed'}
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v2-finalize-") as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(ARCHIVE) as source:
            for name in dict.fromkeys(source.namelist()):
                destination = stage / name; destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(name))
        for source in sorted((ROOT / "prospective").glob("*.py")):
            destination = stage / "prospective" / source.name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        v1_ledger = stage / "prospective_validation" / "ledger.jsonl"; v1_ledger.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "prospective_validation" / "ledger.jsonl", v1_ledger)
        v1_head = stage / "prospective_validation" / "ledger_head.json"
        shutil.copy2(ROOT / "prospective_validation" / "ledger_head.json", v1_head)
        (stage / "reproduce.ps1").write_text(REPRODUCE, encoding="utf-8")
        shutil.copy2(Path(__file__), stage / "prospective_validation_v2" / Path(__file__).name)
        excluded = {"file_manifest.json", "file_manifest.csv", "zip_validation.json"}
        files = [path for path in sorted(stage.rglob("*")) if path.is_file() and path.name not in excluded]
        manifest = [{"relative_path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size,
                     "sha256": digest(path), "description": "V2 review evidence"} for path in files]
        (stage / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with (stage / "file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=["relative_path","bytes","sha256","description"])
            writer.writeheader(); writer.writerows(manifest)
        metadata={"validation_passed":True,"manifest_payload_count":len(manifest),"zip_entry_count":len(manifest)+3}
        (stage / "zip_validation.json").write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as output:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): output.write(path,path.relative_to(stage).as_posix())
    print(json.dumps({"path":str(ARCHIVE.resolve()),"bytes":ARCHIVE.stat().st_size,
                      "sha256":digest(ARCHIVE),"manifest_payload_count":len(manifest),
                      "zip_entry_count":len(manifest)+3},indent=2))


if __name__ == "__main__":
    main()
