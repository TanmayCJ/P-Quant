"""Phase 2: conformal calibration + abstain policy on stored Kronos paths.

python scripts/run_phase2.py --config configs/btc_1h.yaml --root /content/drive/MyDrive/kronos-agent
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.config import dm_horizon, load_config, run_tag
from kronos_agent.data import load_data
from kronos_agent.evaluate import build_eval_frame
from kronos_agent.phase2 import phase2_report
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
    store_df = ResultStore(root / "paths" / tag).load()
    if store_df.empty:
        raise SystemExit(f"no sampled paths for {tag}")

    ev = build_eval_frame(store_df, df, cfg["lookback"], seed=cfg["seed"])
    rep = phase2_report(ev, cfg, delay=dm_horizon(cfg))
    out = root / "reports" / tag
    out.mkdir(parents=True, exist_ok=True)
    for name, t in rep.items():
        if name == "decisions":  # per-origin log for the dashboard
            t.to_parquet(out / f"{name}.parquet", index=False)
        else:
            t.to_csv(out / f"{name}.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 4):
        print("== bias of mean forecast (slope_mu_gap < 0 => reverts to context mean) ==")
        print(rep["bias"].to_string(index=False))
        print("\n== CRPS significance: Kronos dist vs Gaussian (stat<0 & p<0.05 => Kronos better) ==")
        print(rep["crps_tests"].to_string(index=False))
        c = rep["calibration"]
        print("\n== distribution (CRPS, PIT-KS) ==")
        print(c[c["level"].isna()][["period", "base", "method", "crps", "pit_ks"]].to_string(index=False))
        print("\n== intervals ==")
        print(c[c["level"].notna()][["period", "base", "method", "level", "coverage", "width", "winkler",
                                     "spread_skill"]].to_string(index=False))
        print("\n== policy (Kronos) ==")
        print(rep["policy"].to_string(index=False))
        print("\n== AURC (lower = uncertainty more useful) ==")
        print(rep["risk_coverage"].groupby(["period", "confidence"])["aurc"].first().unstack().to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
