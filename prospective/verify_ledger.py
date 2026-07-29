import json
from .canonical import verify_ledger
from .config import LEDGER_HEAD_PATH, LEDGER_PATH
if __name__ == "__main__": print(json.dumps(verify_ledger(LEDGER_PATH,LEDGER_HEAD_PATH),ensure_ascii=False,indent=2))
