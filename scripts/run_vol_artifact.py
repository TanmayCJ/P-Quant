"""Volatility-scaling artifact report: compare the three ways of asking the LLM (CPU only).

python scripts/run_vol_artifact.py --root $ROOT --config configs/vol_artifact/spy.yaml configs/prereg1/qqq.yaml ...
python scripts/run_vol_artifact.py --root $ROOT --config ... --answer-shuffle 200   # + answer-shuffle placebo
python scripts/run_vol_artifact.py --simulate            # synthetic panels only, no data needed
python scripts/run_vol_artifact.py --grid --seeds 50     # false-positive rate by rebound strength x bias
python scripts/run_vol_artifact.py --root $ROOT --config ... --start 2025-01-01   # later dates only
python scripts/run_vol_artifact.py --root $ROOT --config ... --modes labels --controls ret1w ret4w
    # also partial out past returns (short-term reversal check) and test reversal on its own
"""
import argparse
from pathlib import Path

import pandas as pd

from pquant.vol_artifact import BIAS, decompose, past_return, shuffle_answers, simulate, tilt
from pquant.config import load_config
from pquant.prereg_tests import asset_frame
from pquant.data import load_data
from pquant.news_scoring import MODES, llm_dir, llm_tag
from pquant.store import ResultStore

CONTROL_HORIZONS = {"ret1w": 1, "ret4w": 4}  # past return over this many forecast horizons


def controls_for(ctrl: list[str]) -> dict:
    out = {"partial_sigma": ["sigma"]}
    for c in ctrl:
        out[f"partial_{c}"] = [c]
        out[f"partial_sigma_{c}"] = ["sigma", c]
    if len(ctrl) > 1:
        out["partial_all"] = ["sigma", *ctrl]
    return out


def build(cfg, df, llm, variant, mode, ctrl):
    """asset_frame plus the requested past-return columns (rows without enough history dropped)."""
    fr = asset_frame(cfg, df, llm.assign(variant=variant)).assign(mode=mode)
    h = int(cfg["pred_len"])
    for c in ctrl:
        fr[c] = past_return(df, fr["origin_ts"], CONTROL_HORIZONS[c] * h)
    return fr.dropna(subset=ctrl).reset_index(drop=True) if ctrl else fr


def show(per_asset: pd.DataFrame, pooled: pd.DataFrame, title: str):
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(f"== {title}: pooled Fama-MacBeth (signal=raw vs partial_sigma = volatility removed) ==")
        print(pooled.sort_values(["mode", "variant", "signal"]).to_string(index=False))
        cols = ["ic", "ic_partial_sigma", "ic_neg_sigma", "corr_signal_sigma"]
        cols += [c for c in per_asset.columns if c not in cols and c.startswith(("ic_", "corr_signal_"))]
        summ = per_asset.groupby(["mode", "variant"])[cols].mean()
        print(f"\n== {title}: mean across assets ==")
        print(summ.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--variants", nargs="+", default=["real", "shuffled"])
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--answer-shuffle", type=int, default=0, metavar="B",
                    help="B permutations of the real-headline answers (keeps each date's sigma)")
    ap.add_argument("--llm-model", default=None, help="read scores of this model (default: Llama)")
    ap.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                    help="only use forecast dates from this day on (e.g. after a model's training cutoff)")
    ap.add_argument("--controls", nargs="*", default=[], choices=list(CONTROL_HORIZONS),
                    help="also partial out past returns; adds a reversal-only pseudo-mode")
    args = ap.parse_args()

    if args.grid:
        grid(args.seeds)
        if not args.config:
            return

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
    ctrl, controls = args.controls, controls_for(args.controls)
    frames, missing, real_inputs, rev_done = [], [], [], set()
    for p in args.config:
        for mode in args.modes:
            cfg = load_config(p, [f"llm.mode={mode}"] + ([f"llm.model_id={args.llm_model}"] if args.llm_model else []))
            df = load_data(cfg, root / "data")
            for v in args.variants:
                llm = ResultStore(root / "llm" / llm_dir(llm_tag(cfg), cfg, v)).load()
                if args.start and not llm.empty:
                    ts, start = pd.to_datetime(llm["origin_ts"]), pd.Timestamp(args.start)
                    if ts.dt.tz is not None:
                        start = start.tz_localize(ts.dt.tz)
                    llm = llm[ts >= start].reset_index(drop=True)
                if llm.empty:
                    missing.append(f"{cfg['asset']}:{mode}:{v}")
                    continue
                fr = build(cfg, df, llm, v, mode, ctrl)
                frames.append(fr)
                if v == "real":
                    real_inputs.append((cfg, df, llm, mode))
                    if ctrl and cfg["asset"] not in rev_done:  # reversal alone: signal = -past return
                        rev_done.add(cfg["asset"])
                        for c in ctrl:
                            frames.append(fr.assign(mode=f"reversal_{c}", eq=-fr[c]))
    if missing:
        print("missing LLM scores:", ", ".join(missing))
    if not frames:
        raise SystemExit("nothing to report")
    per_asset, pooled = decompose(pd.concat(frames, ignore_index=True), controls)
    name = "artifact" + (f"_{args.llm_model.split('/')[-1]}" if args.llm_model else "")
    name += (f"_from{args.start}" if args.start else "") + (f"_controls-{'-'.join(ctrl)}" if ctrl else "")
    out = root / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"dates per asset: {pooled['n_dates'].max()}" + (f" (from {args.start})" if args.start else ""))
    per_asset.to_csv(out / "per_asset.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    show(per_asset, pooled, "real data")
    if args.answer_shuffle:
        answer_shuffle(real_inputs, args.answer_shuffle, pooled, out, ctrl, controls)
    print(f"\nwrote {out}")


def answer_shuffle(real_inputs, B: int, pooled: pd.DataFrame, out: Path, ctrl=(), controls=None):
    """Pooled test on B answer-shuffled panels; compares each mode's real t with the placebo's."""
    draws = []
    for b in range(B):
        fr = [build(cfg, df, shuffle_answers(llm, seed=b * 1000 + i), "answer_shuffled", mode, list(ctrl))
              for i, (cfg, df, llm, mode) in enumerate(real_inputs)]
        draws.append(decompose(pd.concat(fr, ignore_index=True), controls)[1].assign(draw=b))
    sh = pd.concat(draws, ignore_index=True)
    sh.to_csv(out / "answer_shuffle_draws.csv", index=False)
    real = pooled[pooled["variant"] == "real"].set_index(["mode", "signal"])["t"]
    rows = []
    for (mode, signal), g in sh.groupby(["mode", "signal"]):
        t_real = real.get((mode, signal), float("nan"))
        rows.append({"mode": mode, "signal": signal, "t_real": t_real, "t_placebo_mean": g["t"].mean(),
                     "placebo_rejects_neg": (g["p_one_sided_neg"] < 0.05).mean(),
                     "share_placebo_t_le_real": (g["t"] <= t_real).mean(), "draws": len(g)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "answer_shuffle.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(f"\n== answer-shuffle placebo ({B} draws): answers permuted across dates, sigma kept ==")
        print(summ.to_string(index=False))
        print("A placebo t close to the real t (and frequent rejections) means the signal is carried by the\n"
              "volatility scale, not by the answers. Expect this for volbins and not for fixbins/labels.")


def grid(seeds: int):
    """False-positive rate of the pooled one-sided test with an uninformative LLM, across the
    strength of the volatility-return relation (kappa) and of the response bias (alpha)."""
    rows = []
    for kappa in (0.0, 0.1, 0.25, 0.5):
        for alpha in (0.0, 0.5, 1.0, 1.5):
            pooled = pd.concat([decompose(simulate(kappa=kappa, bias=tilt(BIAS, alpha), seed=s))[1]
                                for s in range(seeds)])
            p = pooled[(pooled["variant"] == "real") & (pooled["signal"] == "raw")]
            for mode, g in p.groupby("mode"):
                rows.append({"kappa": kappa, "alpha": alpha, "mode": mode,
                             "false_positive_rate": (g["p_one_sided_neg"] < 0.05).mean()})
    res = pd.DataFrame(rows)
    out = Path("reports") / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "grid.csv", index=False)
    for mode, g in res.groupby("mode"):
        print(f"== grid, {mode}: false-positive rate, rows kappa (rebound), cols alpha (bias), {seeds} panels ==")
        print(g.pivot(index="kappa", columns="alpha", values="false_positive_rate").round(2).to_string(), "\n")
    print(f"wrote {out / 'grid.csv'}")


if __name__ == "__main__":
    main()
