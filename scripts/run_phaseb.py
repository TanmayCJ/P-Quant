"""Phase B cross-asset report (CPU). A run = model x lookback, given as overrides.

python scripts/run_phaseb.py --root $ROOT --config configs/kronos_b/*.yaml \
    --run kronos64 "lookback=64" --run kronos256 "lookback=256" \
    --run chronos64 "lookback=64;model_family=chronos;model_id=amazon/chronos-t5-small;amp=false"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent.config import dm_horizon, load_config, run_tag
from kronos_agent.data import label_period, load_data, origins_by_window
from kronos_agent.evaluate import build_eval_frame
from kronos_agent.phaseb import asset_metrics, pooled_dm
from kronos_agent.store import ResultStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "OVERRIDES"), required=True,
                    help="name and ';'-separated KEY=VAL overrides")
    ap.add_argument("--k-mode", default="mean_ratio", choices=["mean_ratio", "pooled"],
                    help="EMOS spread estimator: mean_ratio (pre-specified) or pooled (robustness)")
    ap.add_argument("--out", default="phaseb")
    args = ap.parse_args()
    root = Path(args.root)

    rows, pooled, missing, psis = [], [], [], []
    for name, ov in args.run:
        sets = [s for s in ov.split(";") if s]
        losses = []
        for p in args.config:
            cfg = load_config(p, sets)
            s = ResultStore(root / "paths" / run_tag(cfg)).load()
            if s.empty:
                missing.append(f"{name}:{cfg['asset']}")
                continue
            df = load_data(cfg, root / "data")
            # keep exactly this config's origin grid (a store can hold a denser run with the same tag)
            grid = set(np.concatenate(list(origins_by_window(df, cfg).values())).tolist())
            s = s[s["row"].isin(grid)].copy()
            s["period"] = [label_period(pd.Timestamp(t), cfg["windows"]) for t in s["origin_ts"]]
            ev = build_eval_frame(s, df, cfg["lookback"], seed=cfg["seed"])
            row, loss = asset_metrics(ev, cfg, dm_horizon(cfg), k_mode=args.k_mode)
            rows.append({"run": name, "asset": cfg["asset"], **row})
            losses.append(loss)
            psis.append(loss[["date", "psi_slope"]].assign(run=name, asset=cfg["asset"]))
        if losses:
            for col, label in (("d_raw", "raw vs gauss"), ("d_emos", "emos vs gauss")):
                pooled.append({"run": name, "comparison": label, "n_assets": len(losses), **pooled_dm(losses, col)})
    if missing:
        print("missing paths:", ", ".join(missing))
    per = pd.DataFrame(rows)
    pool = pd.DataFrame(pooled)
    out = root / "reports" / args.out
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_asset.csv", index=False)
    pool.to_csv(out / "pooled.csv", index=False)
    if psis:  # per-date slope influence, for date-clustered (Driscoll-Kraay) slope SEs
        pd.concat(psis)[["run", "asset", "date", "psi_slope"]].to_csv(out / "slope_psi.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print("== by run: mean across assets ==")
        agg = per.groupby("run").agg(n_assets=("asset", "count"), slope_gap=("slope_gap", "mean"),
                                     bias=("bias", "mean"), crps_raw=("crps_raw", "mean"),
                                     crps_emos=("crps_emos", "mean"), crps_gauss=("crps_gauss", "mean"),
                                     pit_ks_emos=("pit_ks_emos", "mean"),
                                     share_emos_beats_gauss=("emos_beats_gauss", "mean"))
        print(agg.to_string())
        print("\n== pooled DM on CRPS / gauss CRPS (t<0 => model better than Gaussian) ==")
        print(pool.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
