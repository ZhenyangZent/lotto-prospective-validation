import argparse, json
from .workflow import create_prediction
if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--draw-id",required=True); parser.add_argument("--draw-date",required=True)
    args=parser.parse_args(); print(json.dumps(create_prediction(args.draw_id,args.draw_date),ensure_ascii=False,indent=2,default=str))
