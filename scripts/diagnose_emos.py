"""Why does EMOS make Chronos on ETH much worse? (CPU)

Compares the EMOS spread estimator ('mean_ratio' = current, 'pooled' = ratio of means), with and
without the gap covariate, and a location-only fit (k = 1), and reports how degenerate the
ensembles are (share of forecasts whose sample spread is tiny relative to typical).

python scripts/diagnose_emos.py --root $ROOT --config configs/kronos_b_crypto/eth.yaml \
    --run chronos64 "lookback=64;model_family=chronos;model_id=amazon/chronos-t5-small;amp=false" \
    --run kronos64 "lookback=64"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent import metrics as M
from kronos_agent.conformal import emos_affine
from kronos_agent.config import dm_horizon, load_config, run_tag
from kronos_agent.data import label_period, load_data, origins_by_window
from kronos_agent.evaluate import build_eval_frame
from kronos_agent.store import ResultStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "OVERRIDES"), required=True)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for name, ov in args.run:
        for p in args.config:
            cfg = load_config(p, [s for s in ov.split(";") if s])
            df = load_data(cfg, root / "data")
            s = ResultStore(root / "paths" / run_tag(cfg)).load()
            grid = set(np.concatenate(list(origins_by_window(df, cfg).values())).tolist())
            s = s[s["row"].isin(grid)].copy()
            s["period"] = [label_period(pd.Timestamp(t), cfg["windows"]) for t in s["origin_ts"]]
            ev = build_eval_frame(s, df, cfg["lookback"], seed=cfg["seed"]).sort_values("origin_ts")
            y, gap = ev["r_true"].to_numpy(float), ev["gap"].to_numpy(float)
            X = np.stack(ev["kronos"].to_numpy()).astype(float)
            G = np.stack(ev["gauss"].to_numpy()).astype(float)
            cal, d = cfg["calib"], dm_horizon(cfg)
            m = np.arange(len(y)) >= cal["warmup"] + d - 1
            sd = X.std(1)
            rel = sd / np.median(sd)
            kw = dict(delay=d, window=cal["window"], warmup=cal["warmup"])
            variants = {
                "raw": X,
                "emos_gap_meanratio (reported)": emos_affine(X, y, Z=gap, **kw)[0],
                "emos_gap_pooled": emos_affine(X, y, Z=gap, k_mode="pooled", **kw)[0],
                "emos_nogap_pooled": emos_affine(X, y, k_mode="pooled", **kw)[0],
                "emos_gap_location_only (k=1)": emos_affine(X, y, Z=gap, k_bounds=(1.0, 1.0), **kw)[0],
            }
            cg = M.crps_ensemble(G[m], y[m]).mean()
            base = {"run": name, "asset": cfg["asset"], "n": int(m.sum()),
                    "sd_min/median": float(rel.min()), "share_sd<0.1*median": float((rel < 0.1).mean()),
                    "n_unique_samples_min": int(min(len(np.unique(r)) for r in X))}
            for vname, V in variants.items():
                c = M.crps_ensemble(V[m], y[m]).mean()
                rows.append({**base, "variant": vname, "crps_vs_gauss": c / cg - 1,
                             "pit_ks": M.pit_ks(M.pit(V[m], y[m])),
                             "cov80": float(M.interval_metrics(V[m], y[m], 0.8)["coverage"])})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 4):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
