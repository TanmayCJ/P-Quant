"""Split dashboard_data.json into row datasets for the dashboard page (local, no deps beyond stdlib).

python scripts/dashboard_datasets.py <dashboard_data.json> <out_dir>
Writes prices.json, forecasts.json, llm.json, metrics.json, reversion.json, elicitation_sim.json, preregs.json.
"""
import json
import sys
from pathlib import Path

RUN_LABEL = {"kronos64": ("Kronos-base", 64), "kronos256": ("Kronos-base", 256), "chronos64": ("Chronos-T5-small", 64)}


def main(src: str, out: str):
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prices, fc, llm, metrics = [], [], [], []
    for a in d["assets"]:
        asset = a["asset"].replace("-USDT", "")
        for day, close in a["prices"]:
            prices.append({"asset": asset, "date": day, "close": close})
        for run, m in a["metrics"].items():
            model, lb = RUN_LABEL[run]
            metrics.append({"asset": asset, "run": run, "model": model, "lookback": lb, "n": m["n"],
                            "slope_gap": round(m["slope_gap"], 4), "bias_pct": round(m["bias"] * 100, 3),
                            "crps_raw": round(m["crps_raw"] * 100, 4), "crps_emos": round(m["crps_emos"] * 100, 4),
                            "crps_gauss": round(m["crps_gauss"] * 100, 4), "pit_ks_emos": round(m["pit_ks_emos"], 4),
                            "emos_beats_gauss": bool(m["emos_beats_gauss"])})
        for o in a["origins"]:
            t = o["t"][:10]
            base = {"asset": asset, "t": t, "realized": o["r"], "gap": o["gap"]}
            g = o["fc"].get("gauss")
            if g:
                fc.append({**base, "run": "gauss", "kind": "raw", **dict(zip(("q05", "q25", "q50", "q75", "q95"), g))})
            for run in RUN_LABEL:
                f = o["fc"].get(run)
                if not f:
                    continue
                for kind in ("raw", "emos"):
                    if kind in f:
                        fc.append({**base, "run": run, "kind": kind,
                                   **dict(zip(("q05", "q25", "q50", "q75", "q95"), f[kind]))})
            L = o.get("llm")
            if L:
                row = {"asset": asset, "t": t, "p_up": L["up"], "p_down": L["down"], "p_none": L["none"],
                       "n_heads": L["n_heads"]}
                for i in range(3):
                    h = L["heads"][i] if i < len(L["heads"]) else None
                    row[f"head{i + 1}"] = h["text"] if h else ""
                    row[f"age{i + 1}_h"] = h["age_h"] if h else None
                llm.append(row)

    # Study-level results (from the study's reports: reversion, artifact simulation, pre-registrations)
    # mini and small: exploratory model-size runs (reports/modelsize/size_summary.csv)
    reversion = [
        {"model": "Kronos-mini", "params_m": 4.1, "lookback": 64, "slope_synthetic": -0.05, "slope_real": -0.04},
        {"model": "Kronos-mini", "params_m": 4.1, "lookback": 256, "slope_synthetic": 0.04, "slope_real": 0.02},
        {"model": "Kronos-small", "params_m": 24.7, "lookback": 64, "slope_synthetic": -0.08, "slope_real": -0.19},
        {"model": "Kronos-small", "params_m": 24.7, "lookback": 256, "slope_synthetic": -0.32, "slope_real": -0.55},
        {"model": "Kronos-base", "params_m": 102.3, "lookback": 64, "slope_synthetic": -0.16, "slope_real": -0.19},
        {"model": "Kronos-base", "params_m": 102.3, "lookback": 256, "slope_synthetic": -0.42, "slope_real": -0.70},
        {"model": "Chronos-T5-small", "params_m": 46, "lookback": 64, "slope_synthetic": 0.03, "slope_real": 0.00},
        {"model": "Chronos-T5-small", "params_m": 46, "lookback": 256, "slope_synthetic": 0.01, "slope_real": 0.00},
    ]
    elicitation = [
        {"mode": "Volatility-scaled ranges", "variant": "Real headlines", "rate_pct": 95.0},
        {"mode": "Volatility-scaled ranges", "variant": "Shuffled headlines", "rate_pct": 93.5},
        {"mode": "Volatility-scaled, vol. partialled out", "variant": "Real headlines", "rate_pct": 3.5},
        {"mode": "Volatility-scaled, vol. partialled out", "variant": "Shuffled headlines", "rate_pct": 10.5},
        {"mode": "Fixed ranges", "variant": "Real headlines", "rate_pct": 4.5},
        {"mode": "Fixed ranges", "variant": "Shuffled headlines", "rate_pct": 8.0},
        {"mode": "Direction labels", "variant": "Real headlines", "rate_pct": 3.0},
        {"mode": "Direction labels", "variant": "Shuffled headlines", "rate_pct": 4.0},
    ]
    preregs = [
        {"test": "Pre-registration 1, equities", "signal": "Volatility-scaled ranges", "assets": 8,
         "primary_p": 0.056, "placebo_p": 0.012, "verdict": "Not confirmed"},
        {"test": "Pre-registration 1, crypto", "signal": "Volatility-scaled ranges", "assets": 2,
         "primary_p": 0.727, "placebo_p": 0.008, "verdict": "Not confirmed"},
        {"test": "Pre-registration 2, 5 days", "signal": "Direction labels", "assets": 20,
         "primary_p": 0.600, "placebo_p": 0.876, "verdict": "Not confirmed"},
        {"test": "Pre-registration 2, next day", "signal": "Direction labels", "assets": 20,
         "primary_p": 0.355, "placebo_p": 0.595, "verdict": "Not confirmed (secondary)"},
    ]
    files = {"prices": prices, "forecasts": fc, "llm": llm, "metrics": metrics, "reversion": reversion,
             "elicitation_sim": elicitation, "preregs": preregs}
    for name, rows in files.items():
        p = out / f"{name}.json"
        p.write_text(json.dumps(rows, separators=(",", ":"), ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"{name}: {len(rows)} rows, {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])

