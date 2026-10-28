"""Confirmation study report (PREREGISTRATION.md). CPU only; needs LLM scores from
`run_llm.py --origins-only` for every config.

python scripts/run_confirm.py --root $ROOT \
    --equities configs/confirm/qqq.yaml ... --crypto configs/confirm/btc.yaml configs/confirm/eth.yaml
"""
import argparse
from pathlib import Path

import pandas as pd

from kronos_agent.config import load_config
from kronos_agent.confirm import asset_frame, family_test, holm, per_asset
from kronos_agent.data import load_data
from kronos_agent.newsrun import VARIANTS, llm_dir, llm_tag
from kronos_agent.store import ResultStore


def family_frames(paths: list[str], root: Path, eval_h: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """eval_h: evaluate the same LLM scores against a different return horizon (pre-reg 2, secondary)."""
    frames, missing = [], []
    for p in paths:
        cfg = load_config(p)
        df = load_data(cfg, root / "data")
        ecfg = {**cfg, "pred_len": eval_h} if eval_h else cfg
        for v in VARIANTS:
            llm = ResultStore(root / "llm" / llm_dir(llm_tag(cfg), cfg, v)).load()
            if llm.empty:
                missing.append(f"{cfg['asset']}:{v}")
                continue
            frames.append(asset_frame(ecfg, df, llm.assign(variant=v)))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--equities", nargs="+", default=[])
    ap.add_argument("--crypto", nargs="+", default=[])
    ap.add_argument("--eval-horizon", type=int, default=None,
                    help="score returns over this many bars instead of the config's pred_len")
    ap.add_argument("--out", default="confirm", help="report folder name under reports/")
    args = ap.parse_args()
    root = Path(args.root)

    fam_rows, asset_rows = [], []
    for fam, paths in (("equities", args.equities), ("crypto", args.crypto)):
        if not paths:
            continue
        frames, missing = family_frames(paths, root, args.eval_horizon)
        if missing:
            print(f"[{fam}] missing LLM scores: {', '.join(missing)}")
        if frames.empty:
            continue
        asset_rows.append(per_asset(frames).assign(family=fam))
        for v in VARIANTS:
            fam_rows.append({"family": fam, **family_test(frames, v)})

    fam_df = pd.DataFrame(fam_rows)
    primary = fam_df[fam_df["variant"] == "real"].dropna(subset=["p_one_sided"])
    adj = holm(dict(zip(primary["family"], primary["p_one_sided"])))
    fam_df["p_holm"] = [adj.get(f) if v == "real" else float("nan") for f, v in zip(fam_df["family"], fam_df["variant"])]
    fam_df["H1_confirmed"] = fam_df["p_holm"] < 0.05

    out = root / "reports" / args.out
    out.mkdir(parents=True, exist_ok=True)
    fam_df.to_csv(out / "family_tests.csv", index=False)
    assets = pd.concat(asset_rows, ignore_index=True) if asset_rows else pd.DataFrame()
    assets.to_csv(out / "per_asset.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print("== PRIMARY: family tests (variant=real decides; Holm across families) ==")
        print(fam_df.to_string(index=False))
        print("\n== per-asset IC (secondary) ==")
        print(assets.sort_values(["family", "variant", "asset"]).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
