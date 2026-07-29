import json
from .workflow import status_summary
if __name__ == "__main__": print(json.dumps(status_summary(),ensure_ascii=False,indent=2))
