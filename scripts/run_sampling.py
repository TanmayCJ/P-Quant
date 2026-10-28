"""Sample forecast paths for every origin in one or more configs (resumable; model loaded once).

python scripts/run_sampling.py --config configs/btc_1h.yaml --root /content/drive/MyDrive/kronos-agent \
    --kronos-path /content/Kronos [--limit 3] [--set amp=true --set n_paths=32]
python scripts/run_sampling.py --config configs/kronos_b/*.yaml --set lookback=64 ...      # many assets
python scripts/run_sampling.py --config ... --set model_family=chronos --set model_id=amazon/chronos-t5-small --set amp=false
"""
import argparse
import json
import subprocess
from pathlib import Path

from kronos_agent.baselines import make_sampler
from kronos_agent.config import load_config, run_tag
from kronos_agent.data import load_data
from kronos_agent.pipeline import run_sampling
from kronos_agent.store import ResultStore


def git_rev(path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--root", required=True, help="output root (Drive folder on Colab)")
    ap.add_argument("--kronos-path", default=None, help="clone of shiyu-coder/Kronos (Kronos models only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override config, e.g. --set amp=true --set n_paths=32")
    args = ap.parse_args()
    root = Path(args.root)

    sampler, loaded_for = None, None
    for config in args.config:
        cfg = load_config(config, args.set)
        df = load_data(cfg, root / "data")
        store = ResultStore(root / "paths" / run_tag(cfg))
        meta = {**cfg, "kronos_commit": git_rev(args.kronos_path) if args.kronos_path else None,
                "n_bars": len(df), "data_first": str(df.index[0]), "data_last": str(df.index[-1])}
        (store.root / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        key = (cfg.get("model_family"), cfg["model_id"], cfg["tokenizer_id"], cfg["amp"], cfg["max_context"])
        if sampler is None or loaded_for != key:
            sampler, loaded_for = make_sampler(cfg, args.kronos_path, args.device), key
            print(f"model={cfg['model_id']} device={sampler.device()} amp={cfg['amp']}")
        print(f"out={store.root}")
        run_sampling(cfg, df, sampler, store, limit=args.limit)


if __name__ == "__main__":
    main()
