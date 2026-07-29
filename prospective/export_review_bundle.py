import argparse, json
from .workflow import export_review_bundle, verify_bundle
if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--verify-only")
    args=parser.parse_args(); print(json.dumps(verify_bundle(args.verify_only) if args.verify_only else export_review_bundle(),ensure_ascii=False,indent=2))
