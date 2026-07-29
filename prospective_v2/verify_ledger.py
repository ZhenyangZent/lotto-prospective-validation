import json
from .workflow import verify_v2_ledger
if __name__ == "__main__": print(json.dumps(verify_v2_ledger(),ensure_ascii=False,indent=2))
