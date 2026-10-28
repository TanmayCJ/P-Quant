"""Phase 3b: news-likelihood fusion report (CPU).

python scripts/run_phase3.py --config configs/spy_1d.yaml --root $ROOT
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.config import dm_horizon, load_config, run_tag
from kronos_agent.data import label_period, load_data
from kronos_agent.evaluate import build_eval_frame
from kronos_agent.newsrun import VARIANTS
from kronos_agent.phase3 import phase3_report
from kronos_agent.store import ResultStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    root = Path(args.root)
    df = load_data(cfg, root / "data")
    tag = run_tag(cfg)
    s = ResultStore(root / "paths" / tag).load()
    s["period"] = [label_period(pd.Timestamp(t), cfg["windows"]) for t in s["origin_ts"]]
    s = s[s["period"].notna()]
    ev = build_eval_frame(s, df, cfg["lookback"], seed=cfg["seed"])

    parts = []
    for v in VARIANTS:
        d = ResultStore(root / "llm" / f"{tag}_{v}").load()
        if not d.empty:
            parts.append(d)
    if not parts:
        raise SystemExit("no LLM scores found; run scripts/run_llm.py first")
    rep = phase3_report(ev, pd.concat(parts, ignore_index=True), cfg, dm_horizon(cfg))

    out = root / "reports" / tag
    out.mkdir(parents=True, exist_ok=True)
    for name, t in rep.items():
        t.to_csv(out / f"phase3_{name}.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 40, "display.precision", 4):
        print("== LLM signal (ic_EQ = IC of the LLM's expected return; ic_p = HAC p-value) ==")
        print(rep["llm"].to_string(index=False))
        print("\n== fusion vs prior (dm_stat<0 & dm_p<0.05 => news improves CRPS) ==")
        f = rep["fusion"].sort_values(["method", "period", "prior", "variant", "q_mode"])
        cols = ["method", "period", "prior", "variant", "q_mode", "n", "crps_prior", "crps_fused", "dm_stat",
                "dm_p", "lam_mean", "lam_neg_frac", "pit_ks_fused", "cov80_fused", "ic_prior", "ic_fused",
                "dir_prior", "dir_fused"]
        print(f[cols].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
