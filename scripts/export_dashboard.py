"""Export saved results to reports/dashboard/dashboard_data.json for the dashboard (CPU, a few min).

python scripts/export_dashboard.py --root $ROOT
"""
import argparse
import json
import time
from pathlib import Path

from kronos_agent.export import export_asset, llm_config_for

CODE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    configs = sorted((CODE / "configs" / "kronos_b").glob("*.yaml")) + [CODE / "configs" / "kronos_b_crypto" / "eth.yaml"]
    assets, t0 = [], time.time()
    for p in configs:
        import yaml

        asset = yaml.safe_load(p.read_text())["asset"]
        a = export_asset(str(p), root, llm_config_for(asset, CODE))
        if a is None:
            print(f"  {asset}: no forecasts found, skipped")
            continue
        n_llm = sum("llm" in o for o in a["origins"])
        print(f"  {asset}: {len(a['origins'])} origins, runs {sorted(a['metrics'])}, LLM on {n_llm}")
        assets.append(a)
    out = root / "reports" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "dashboard_data.json"
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "quantiles": [5, 25, 50, 75, 95],
               "assets": assets}
    f.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"wrote {f} ({f.stat().st_size / 1e6:.1f} MB, {len(assets)} assets, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
