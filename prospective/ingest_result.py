import argparse, json
from .workflow import ingest_result
if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--draw-id",required=True)
    args=parser.parse_args(); print(json.dumps(ingest_result(args.draw_id),ensure_ascii=False,indent=2))
