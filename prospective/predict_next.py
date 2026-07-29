import argparse, json
from .workflow import predict_next
if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--draw-id",required=True); parser.add_argument("--draw-date",required=True)
    args=parser.parse_args(); print(json.dumps(predict_next(args.draw_id,args.draw_date),ensure_ascii=False,indent=2))
