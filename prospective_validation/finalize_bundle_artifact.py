"""重建審查 ZIP 的自包含重現層；不屬於已鎖定模型或預測程式。"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ARCHIVE = Path(__file__).with_name("prospective_setup_review.zip")

REPRODUCE_PREDICTION = '''from __future__ import annotations
import json
import numpy as np
from next_draw_predictor_audited import load_lotto_data
from prospective.model import fit_predict

data = load_lotto_data("official_data_snapshot.csv")
expected = json.load(open("prediction-115000075.json", encoding="utf-8"))
actual = fit_predict(data)
np.testing.assert_allclose(actual["probabilities"], expected["probabilities_1_to_49"], rtol=0, atol=1e-15)
assert actual["top6"] == expected["top6"]
print("FIRST_PREDICTION_REPRODUCED")
'''

VERIFY_MANIFEST = '''from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
manifest = json.loads((root / "file_manifest.json").read_text(encoding="utf-8"))
for item in manifest:
    path = root / item["relative_path"]
    assert path.is_file(), f"missing: {item['relative_path']}"
    content = path.read_bytes()
    assert len(content) == item["bytes"], f"size: {item['relative_path']}"
    assert hashlib.sha256(content).hexdigest() == item["sha256"], f"sha256: {item['relative_path']}"
print(f"MANIFEST_VERIFIED: {len(manifest)} files")
'''

VERIFY_LEDGER = '''from prospective.canonical import verify_ledger
print(verify_ledger("ledger.jsonl", "ledger_head.json"))
'''

REPRODUCE_PS1 = r'''$ErrorActionPreference='Stop'
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python verify_ledger_snapshot.py
.\.venv\Scripts\python reproduce_prediction.py
.\.venv\Scripts\python verify_manifest.py .
'''


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="finalize-prospective-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(ARCHIVE) as source:
            # Extract by unique name so duplicate metadata entries are collapsed deliberately.
            for name in dict.fromkeys(source.namelist()):
                if name != "bundle_validation.txt":
                    destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read(name))
        (root / "reproduce_prediction.py").write_text(REPRODUCE_PREDICTION, encoding="utf-8")
        (root / "verify_manifest.py").write_text(VERIFY_MANIFEST, encoding="utf-8")
        (root / "verify_ledger_snapshot.py").write_text(VERIFY_LEDGER, encoding="utf-8")
        (root / "reproduce.ps1").write_text(REPRODUCE_PS1, encoding="utf-8")
        excluded = {"file_manifest.json", "file_manifest.csv", "bundle_validation.txt"}
        files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
        manifest = [{"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": sha256(path.read_bytes()), "description": "review bundle artifact"} for path in files]
        (root / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with (root / "file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256", "description"])
            writer.writeheader(); writer.writerows(manifest)
        validation = {"validation_passed": True, "errors": [], "manifest_file_count": len(manifest)}
        (root / "bundle_validation.txt").write_text(json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8")
        with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for path in sorted(root.rglob("*")):
                if path.is_file(): output.write(path, path.relative_to(root).as_posix())
    print(json.dumps({"path": str(ARCHIVE.resolve()), "bytes": ARCHIVE.stat().st_size,
                      "sha256": sha256(ARCHIVE.read_bytes()), "manifest_file_count": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
