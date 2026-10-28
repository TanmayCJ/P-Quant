"""Phase A report: volatility-scaling artifact across elicitation modes (CPU).

python scripts/run_artifact.py --root $ROOT --config configs/artifact/spy.yaml configs/confirm/qqq.yaml ...
python scripts/run_artifact.py --simulate            # synthetic panels only, no data needed
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.artifact import decompose, simulate
from kronos_agent.config import load_config
from kronos_agent.confirm import asset_frame
from kronos_agent.data import load_data
from kronos_agent.newsrun import MODES, llm_dir, llm_tag
from kronos_agent.store import ResultStore


def show(per_asset: pd.DataFrame, pooled: pd.DataFrame, title: str):
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(f"== {title}: pooled Fama-MacBeth (signal=raw vs partial_sigma = volatility removed) ==")
        print(pooled.sort_values(["mode", "variant", "signal"]).to_string(index=False))
        summ = per_asset.groupby(["mode", "variant"])[["ic", "ic_partial_sigma", "ic_neg_sigma",
                                                        "corr_signal_sigma"]].mean()
        print(f"\n== {title}: mean across assets ==")
        print(summ.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--variants", nargs="+", default=["real", "shuffled"])
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()

    if args.simulate:
        for info in (0.0, 0.3):
            pooled = pd.concat([decompose(simulate(info=info, seed=s))[1].assign(seed=s)
                                for s in range(args.seeds)])
            summ = pooled.groupby(["mode", "variant", "signal"]).agg(
                mean_t=("t", "mean"), rejects_neg=("p_one_sided_neg", lambda x: (x < 0.05).mean()),
                rejects_two_sided=("p_two_sided", lambda x: (x < 0.05).mean()))
            print(f"== simulation, info={info}, {args.seeds} panels of 9 assets ==")
            print(summ.round(3).to_string(), "\n")
        if not args.config:
            return

    root = Path(args.root)
    frames, missing = [], []
    for p in args.config:
        for mode in args.modes:
            cfg = load_config(p, [f"llm.mode={mode}"])
            df = load_data(cfg, root / "data")
            for v in args.variants:
                llm = ResultStore(root / "llm" / llm_dir(llm_tag(cfg), cfg, v)).load()
                if llm.empty:
                    missing.append(f"{cfg['asset']}:{mode}:{v}")
                    continue
                frames.append(asset_frame(cfg, df, llm.assign(variant=v)).assign(mode=mode))
    if missing:
        print("missing LLM scores:", ", ".join(missing))
    if not frames:
        raise SystemExit("nothing to report")
    per_asset, pooled = decompose(pd.concat(frames, ignore_index=True))
    out = root / "reports" / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    per_asset.to_csv(out / "per_asset.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    show(per_asset, pooled, "real data")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
