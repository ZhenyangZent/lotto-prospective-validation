import json
from .workflow import report_results
if __name__ == "__main__": print(json.dumps(report_results(),ensure_ascii=False,indent=2))
