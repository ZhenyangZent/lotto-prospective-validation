$ErrorActionPreference='Stop'
& .\.audit_venv\Scripts\python.exe audit_stage2\run_stage2.py actual
& .\.audit_venv\Scripts\python.exe audit_stage2\run_stage2.py null --runs 1000 --workers 4
& .\.audit_venv\Scripts\python.exe audit_stage2\run_stage2.py simplified --runs 10000 --workers 4
& .\.audit_venv\Scripts\python.exe audit_stage2\run_stage2.py stability
& .\.audit_venv\Scripts\python.exe audit_stage2\finalize_stage2.py
