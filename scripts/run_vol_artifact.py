"""Volatility-scaling artifact report: compare the three ways of asking the LLM (CPU only).

python scripts/run_vol_artifact.py --root $ROOT --config configs/vol_artifact/spy.yaml configs/prereg1/qqq.yaml ...
python scripts/run_vol_artifact.py --root $ROOT --config ... --answer-shuffle 200   # + answer-shuffle placebo
python scripts/run_vol_artifact.py --simulate            # synthetic panels only, no data needed
python scripts/run_vol_artifact.py --grid --seeds 50     # false-positive rate by rebound strength x bias
python scripts/run_vol_artifact.py --root $ROOT --config ... --start 2025-01-01   # later dates only
python scripts/run_vol_artifact.py --root $ROOT --config ... --modes labels --controls ret1w ret4w [--tag prereg2]
    # also partial out past returns (short-term reversal check) and test reversal on its own
python scripts/run_vol_artifact.py --root $ROOT --config ... --events   # volatility-shock weeks
python scripts/run_vol_artifact.py --root $ROOT --config ... --modes volbins --lean   # predict the fake signal
python scripts/run_vol_artifact.py --grid-predict    # formula vs simulation grid (no data needed)
python scripts/run_vol_artifact.py --root $ROOT --config ... --pnl 1000   # backtest P&L + answer-shuffle placebo
python scripts/run_vol_artifact.py --pnl-sim         # does an uninformative model show backtest "alpha"? (no data)
python scripts/run_vol_artifact.py --grid --grid-predict --seeds 200 --T 115   # grid at the real sample length
python scripts/run_vol_artifact.py --root $ROOT --config ... --modes volbins --grid-position   # real sample on the grid
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pquant.vol_artifact import (BIAS, decompose, drop_test, event_dates, past_return, pooled_series,
                                  shock_dates, shuffle_answers, simulate, tilt, vol_rank,
                                  grid_predict, lean_table, predict_pooled, pnl_sim, pnl_table,
                                  grid_position, sim_lean_ratios)
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
    ap.add_argument("--quant-bits", type=int, default=4, choices=(4, 8, 16),
                    help="read scores made at this precision (default 4-bit)")
    ap.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                    help="only use forecast dates from this day on (e.g. after a model's training cutoff)")
    ap.add_argument("--events", action="store_true",
                    help="does the result live in volatility-shock weeks? (drop them vs random weeks)")
    ap.add_argument("--shock-quantile", type=float, default=0.9)
    ap.add_argument("--lean", action="store_true", help="predict the fake signal from the answers' lean")
    ap.add_argument("--grid-predict", action="store_true", help="formula's false-alarm rate per grid cell")
    ap.add_argument("--T", type=int, default=400, help="periods per simulated panel for --grid / --grid-predict")
    ap.add_argument("--grid-position", action="store_true",
                    help="place the real sample on the simulation grid (kappa_hat, alpha-equivalent; T=115)")
    ap.add_argument("--pnl", type=int, default=None, metavar="B",
                    help="backtest P&L of the forecasts, with B answer-shuffle placebo draws (0 = none)")
    ap.add_argument("--pnl-sim", action="store_true", help="P&L of an uninformative model, simulation grid")
    ap.add_argument("--tag", default=None, help="suffix for the output folder (keeps runs on other assets apart)")
    ap.add_argument("--controls", nargs="*", default=[], choices=list(CONTROL_HORIZONS),
                    help="also partial out past returns; adds a reversal-only pseudo-mode")
    args = ap.parse_args()

    if args.grid_predict:
        res = grid_predict(T=args.T)
        Path("reports/artifact").mkdir(parents=True, exist_ok=True)
        res.to_csv(f"reports/artifact/grid_predict{_tsuffix(args.T)}.csv", index=False)
        print(f"== formula: predicted false-alarm rate, volbins, T={args.T}, rows kappa, cols alpha ==")
        print(res.pivot(index="kappa", columns="alpha", values="false_alarm_predicted").round(2).to_string())
        if not args.config and not args.grid:
            return
    if args.pnl_sim:
        res = pnl_sim()
        Path("reports/artifact").mkdir(parents=True, exist_ok=True)
        res.to_csv("reports/artifact/pnl_sim.csv", index=False)
        with pd.option_context("display.width", 200, "display.precision", 3):
            print("== simulation: vol-target backtest of an uninformative model (volbins, T=115, 200 panels) ==")
            print(res.to_string(index=False))
        if not args.config and not args.grid:
            return
    if args.grid:
        grid(args.seeds, args.T)
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
            cfg = load_config(p, [f"llm.mode={mode}", f"llm.quant_bits={args.quant_bits}"]
                              + ([f"llm.model_id={args.llm_model}"] if args.llm_model else []))
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
    frames = pd.concat(frames, ignore_index=True)
    per_asset, pooled = decompose(frames, controls)
    name = "artifact" + (f"_{args.llm_model.split('/')[-1]}" if args.llm_model else "")
    name += f"_{args.quant_bits}bit" if args.quant_bits != 4 else ""
    name += (f"_from{args.start}" if args.start else "") + (f"_controls-{'-'.join(ctrl)}" if ctrl else "")
    name += f"_{args.tag}" if args.tag else ""
    out = root / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"dates per asset: {pooled['n_dates'].max()}" + (f" (from {args.start})" if args.start else ""))
    per_asset.to_csv(out / "per_asset.csv", index=False)
    pooled.to_csv(out / "pooled.csv", index=False)
    show(per_asset, pooled, "real data")
    if args.lean:
        lean(frames, pooled, out)
    if args.grid_position:
        position(frames, out)
    if args.pnl is not None:
        pnl(frames, out, args.pnl)
    if args.events:
        events(frames, controls, args.shock_quantile, out)
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


def events(frames: pd.DataFrame, controls: dict, q: float, out: Path, B: int = 1000):
    """Drop volatility-shock weeks (top 1-q by mean vol rank) and named events; compare with
    dropping the same number of random weeks."""
    series = pooled_series(frames, controls)
    vr = vol_rank(frames)
    named = event_dates(series.index)
    top = f"top{round((1 - q) * 100)}pct_vol"
    sets = {top: shock_dates(frames, q), **named,
            "both_events": named["yen_carry_2024-08"].union(named["tariffs_2025-04"])}
    rows = []
    for (mode, variant, signal), s in series.items():
        if signal not in ("raw", "partial_sigma"):
            continue
        for name, drop in sets.items():
            rows.append({"mode": mode, "variant": variant, "signal": signal, "dropped": name,
                         **drop_test(s, drop, B=B)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "events.csv", index=False)
    wide = series.copy()
    wide.columns = ["/".join(c) for c in wide.columns]
    wide.assign(vol_rank=vr).to_csv(out / "per_date.csv")
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 3):
        for name, d in sets.items():
            print(f"{name}: {len(d)} dates: " + ", ".join(str(x.date()) for x in d))
        print(f"\n== volatility-shock weeks: pooled t with all dates vs without them ({B} random drops) ==")
        print(res.drop(columns="n_dates").to_string(index=False))
        print("p_random small = dropping these weeks moves t toward 0 far more than dropping random weeks.")
        key = ("volbins", "shuffled", "raw")
        if key in series:
            worst = series[key].nsmallest(8)
            print("\n== dates pushing volbins/shuffled most negative ==")
            print(pd.DataFrame({"contribution": worst, "vol_rank": vr.reindex(worst.index)}).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = [k for k in [("volbins", "shuffled", "raw"), ("volbins", "real", "raw"),
                        ("fixbins", "real", "raw"), ("labels", "real", "raw")] if k in series]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2.2 * len(keys)), sharex=True, squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        ax.bar(series.index, series[k], width=4, color="#444")
        for d in sets[top]:
            ax.axvspan(d - pd.Timedelta(days=3), d + pd.Timedelta(days=3), color="#f2a65a", alpha=0.35, lw=0)
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_ylabel("/".join(k[:2]), fontsize=8)
    fig.suptitle("Per-week contribution to the pooled test (orange = high-volatility weeks)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "events.png", dpi=150)
    print(f"plot: {out / 'events.png'}")


def pnl(frames: pd.DataFrame, out: Path, B: int):
    """Check 1: backtest P&L of the model's forecasts, vs buy-and-hold and the volatility-managed long."""
    f = frames[~frames["mode"].str.startswith("reversal_")]
    res = pnl_table(f, B=B)
    res.to_csv(out / "pnl.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(f"\n== trading P&L (weekly, no costs; answer-shuffle placebo {B} draws) ==")
        print(res.to_string(index=False))
        print("Diagnostic, not a strategy. 'Volatility timing' = real Sharpe inside the placebo's 5-95% range and\n"
              "|t_alpha_bh_vm| < 1.96. Negative alpha = following the model lost money.")


def lean(frames: pd.DataFrame, pooled: pd.DataFrame, out: Path):
    """Formula check on real data: predicted vs observed fake signal for volatility-scaled boxes."""
    f = frames[frames["mode"] == "volbins"]
    if f.empty:
        print("--lean needs volbins scores")
        return
    lt, pp = lean_table(f), predict_pooled(f, pooled)
    lt.to_csv(out / "lean_per_asset.csv", index=False)
    pp.to_csv(out / "lean.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\n== lean formula, per asset (volbins) ==")
        print(lt.to_string(index=False))
        print("\n== lean formula, pooled: predicted vs observed ==")
        print(pp.to_string(index=False))
        print("Compare t_predicted with the answer-shuffle placebo's mean t (the uninformative benchmark).")


def _tsuffix(T: int) -> str:
    return "" if T == 400 else f"_T{T}"


def position(frames: pd.DataFrame, out: Path):
    """Where the real sample sits on the simulation grid (rule H2 in the analysis plan)."""
    res = pd.DataFrame([grid_position(frames, sim_lean_ratios(T=115))])
    res.to_csv(out / "grid_position.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print("\n== real sample on the simulation grid (volbins, real headlines) ==")
        print(res.to_string(index=False))


def grid(seeds: int, T: int = 400):
    """False-positive rate of the pooled one-sided test with an uninformative LLM, across the
    strength of the volatility-return relation (kappa) and of the response bias (alpha).
    One row per mode and signal (raw, and with sigma partialled out)."""
    rows = []
    for kappa in (0.0, 0.1, 0.25, 0.5):
        for alpha in (0.0, 0.5, 1.0, 1.5):
            pooled = pd.concat([decompose(simulate(T=T, kappa=kappa, bias=tilt(BIAS, alpha), seed=s))[1]
                                for s in range(seeds)])
            p = pooled[pooled["variant"] == "real"]
            for (mode, signal), g in p.groupby(["mode", "signal"]):
                rows.append({"kappa": kappa, "alpha": alpha, "mode": mode, "signal": signal,
                             "false_positive_rate": (g["p_one_sided_neg"] < 0.05).mean()})
    res = pd.DataFrame(rows)
    out = Path("reports") / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"grid{_tsuffix(T)}.csv"
    res.to_csv(path, index=False)
    for (mode, signal), g in res.groupby(["mode", "signal"]):
        print(f"== grid T={T}, {mode} / {signal}: false-positive rate, rows kappa, cols alpha, {seeds} panels ==")
        print(g.pivot(index="kappa", columns="alpha", values="false_positive_rate").round(2).to_string(), "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
