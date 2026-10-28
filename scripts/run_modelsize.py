"""Model-size study report (CPU): does Kronos' window-mean reversion depend on model size?

Run after run_phaseb.py --out modelsize with runs named <size><lookback> (base64, small256, chronos64, ...)
and run_synthetic.py for each size. t/se are across assets (naive: assets share dates); t_dk/p_dk use
Driscoll-Kraay SEs (per-date influence summed over assets, Newey-West over dates; L10/L20 = sensitivity).

python scripts/run_modelsize.py --root $ROOT
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kronos_agent import metrics as M
from kronos_agent.phaseb import slope_dk, slope_influence, summarize_synthetic

# chronos: comparison model (its "diff_vs_base" is Chronos minus Kronos-base)
SIZES = {"mini": ("Kronos-mini", 4.1), "small": ("Kronos-small", 24.7), "base": ("Kronos-base", 102.3),
         "chronos": ("Chronos-T5-small", 46.0)}


def _t(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(se), float(x.mean() / se) if se else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--report", default="modelsize")
    args = ap.parse_args()
    root = Path(args.root)
    per = pd.read_csv(root / "reports" / args.report / "per_asset.csv")
    syn_dir = root / "reports" / "synthetic"
    f_psi = root / "reports" / args.report / "slope_psi.csv"
    psi = pd.read_csv(f_psi, parse_dates=["date"]) if f_psi.exists() else None
    if psi is None:
        print(f"no {f_psi}: rerun run_phaseb.py for date-clustered slope SEs")

    rows = []
    for size, (model, params) in SIZES.items():
        for L in (64, 256):
            g = per[per["run"] == f"{size}{L}"]
            if g.empty:
                continue
            m, se, t = _t(g["slope_gap"])
            f = syn_dir / f"{model}_L{L}.csv"
            syn = float("nan")
            if f.exists():
                s = summarize_synthetic(pd.read_csv(f))
                syn = float(s.loc[s["drift"].astype(str) == "all", "slope_fc_gap"].iloc[0])
            row = {"model": model, "params_M": params, "lookback": L, "n_assets": len(g), "slope_real": m,
                   "se": se, "t": t, "slope_synthetic": syn,
                   "crps_raw_vs_gauss_pct": float((g["crps_raw"] / g["crps_gauss"] - 1).mean() * 100),
                   "crps_emos_vs_gauss_pct": float((g["crps_emos"] / g["crps_gauss"] - 1).mean() * 100)}
            # paired across assets: this size's slope minus Kronos-base's at the same lookback
            b = per[per["run"] == f"base{L}"].set_index("asset")["slope_gap"]
            d = g.set_index("asset")["slope_gap"] - b
            row["diff_vs_base"], _, row["t_diff_vs_base"] = _t(d) if size != "base" else (0.0, 0.0, float("nan"))
            if psi is not None:  # same estimates, SEs clustered by date (stocks share dates)
                u = slope_influence(psi[psi["run"] == f"{size}{L}"])
                dk = slope_dk(u)
                row.update(n_dates=dk["n_dates"], lag=dk["lag_auto"], se_dk=dk["se_dk"], t_dk=m / dk["se_dk"],
                           p_dk=M._t_sf_two_sided(m / dk["se_dk"], dk["n_dates"] - 1),
                           t_dk_L10=m / dk["se_dk_L10"], t_dk_L20=m / dk["se_dk_L20"])
                if size != "base":
                    ub = slope_influence(psi[psi["run"] == f"base{L}"])
                    ud = u.sub(ub, fill_value=0.0)
                    dd = slope_dk(ud)
                    td = row["diff_vs_base"] / dd["se_dk"]
                    row.update(t_dk_diff_vs_base=td, p_dk_diff_vs_base=M._t_sf_two_sided(td, dd["n_dates"] - 1),
                               t_dk_diff_L20=row["diff_vs_base"] / dd["se_dk_L20"])
            rows.append(row)
    out = pd.DataFrame(rows)
    path = root / "reports" / args.report / "size_summary.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("== reversion slope (forecast mean on gap; 0 = no reversion) by model size ==")
        print(out.drop(columns=[c for c in out.columns if "dk" in c or c in ("n_dates", "lag")]).to_string(index=False))
        if "t_dk" in out:
            cols = ["model", "lookback", "slope_real", "t", "n_dates", "lag", "se_dk", "t_dk", "p_dk", "t_dk_L10",
                    "t_dk_L20", "t_dk_diff_vs_base", "p_dk_diff_vs_base", "t_dk_diff_L20"]
            print("\n== date-clustered (Driscoll-Kraay) slope inference ==")
            print(out[[c for c in cols if c in out]].to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
