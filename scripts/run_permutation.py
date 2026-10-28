"""Permutation p-values next to the Newey-West ones for the pooled IC tests (CPU).

python scripts/run_permutation.py --root $ROOT --name prereg1_equities --mode volbins --config $EQ
python scripts/run_permutation.py --root $ROOT --name artifact --mode volbins fixbins labels --config $ALL --partial
python scripts/run_permutation.py --root $ROOT --name prereg2 --mode labels --config $C2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent.artifact import _resid
from kronos_agent.config import load_config
from kronos_agent.confirm import asset_frame, family_test, perm_test
from kronos_agent.data import load_data
from kronos_agent.newsrun import llm_dir, llm_tag
from kronos_agent.store import ResultStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--mode", nargs="+", default=["volbins"])
    ap.add_argument("--variants", nargs="+", default=["real", "shuffled"])
    ap.add_argument("--partial", action="store_true", help="also test the signal with volatility partialled out")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for mode in args.mode:
        frames = []
        for p in args.config:
            cfg = load_config(p, [f"llm.mode={mode}"])
            df = load_data(cfg, root / "data")
            for v in args.variants:
                llm = ResultStore(root / "llm" / llm_dir(llm_tag(cfg), cfg, v)).load()
                if not llm.empty:
                    frames.append(asset_frame(cfg, df, llm.assign(variant=v)))
        if not frames:
            print(f"[{mode}] no LLM scores found")
            continue
        f = pd.concat(frames, ignore_index=True)
        if args.partial:
            f["eq_partial"] = np.nan
            for _, idx in f.groupby(["asset", "variant"]).groups.items():
                g = f.loc[idx]
                f.loc[idx, "eq_partial"] = _resid(g["eq"].to_numpy(float), g["sigma"].to_numpy(float))
        for v in args.variants:
            for col in ["eq"] + (["eq_partial"] if args.partial else []):
                hac = family_test(f.assign(eq=f[col]), v)
                pt = perm_test(f, v, B=args.B, col=col)
                rows.append({"study": args.name, "mode": mode, "variant": v,
                             "signal": "raw" if col == "eq" else "partial_sigma", "n_assets": hac["n_assets"],
                             "n_dates": hac["n_dates"], "mean_d": hac["mean_d"], "hac_t": hac["t"],
                             "hac_p_one_sided": hac["p_one_sided"], "perm_p_one_sided": pt["p_one_sided_neg"],
                             "perm_p_two_sided": pt["p_two_sided"]})
    out = pd.DataFrame(rows)
    d = root / "reports" / "permutation"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"{args.name}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 4):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
