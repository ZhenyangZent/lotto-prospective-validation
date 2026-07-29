import json
from .workflow import freeze_experiment
if __name__ == "__main__": print(json.dumps(freeze_experiment(), ensure_ascii=False, indent=2))
