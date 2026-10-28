"""Phase B mechanism test: forecast random walks with known drift (GPU).

python scripts/run_synthetic.py --root $ROOT --kronos-path $KRONOS --lookback 64 256
python scripts/run_synthetic.py --root $ROOT --model-family chronos --model-id amazon/chronos-t5-small --lookback 64 256
python scripts/run_synthetic.py --root $ROOT --kronos-path $KRONOS --model-id NeoQuasar/Kronos-mini --tokenizer-id NeoQuasar/Kronos-Tokenizer-2k
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.baselines import make_sampler
from kronos_agent.config import DEFAULTS
from kronos_agent.phaseb import run_synthetic, summarize_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kronos-path", default=None)
    ap.add_argument("--model-family", default="kronos")
    ap.add_argument("--model-id", default=DEFAULTS["model_id"])
    ap.add_argument("--tokenizer-id", default=DEFAULTS["tokenizer_id"])
    ap.add_argument("--lookback", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--n-series", type=int, default=40)
    args = ap.parse_args()

    cfg = {**DEFAULTS, "model_family": args.model_family, "model_id": args.model_id,
           "tokenizer_id": args.tokenizer_id, "amp": args.model_family == "kronos"}
    sampler = make_sampler(cfg, args.kronos_path)
    out = Path(args.root) / "reports" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    name = args.model_id.split("/")[-1]
    for L in args.lookback:
        print(f"== {name}, lookback {L} ==")
        res = run_synthetic(sampler, L, n_series=args.n_series)
        res.to_csv(out / f"{name}_L{L}.csv", index=False)
        with pd.option_context("display.width", 200, "display.precision", 4):
            print(summarize_synthetic(res).to_string(index=False), "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
