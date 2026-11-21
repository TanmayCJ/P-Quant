"""Comparison foundation model with the same sampling interface as KronosSampler.

Chronos (T5, sample-based) forecasts the close series only; the other OHLC fields are
filled with the close so the pipeline, store and evaluator work unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import OHLCV
from .sampler import _seed_all


class ChronosSampler:
    def __init__(self, pipe, model_id: str, max_context: int = 512):
        self.pipe = pipe
        self.model_id = model_id
        self.max_context = max_context

    @classmethod
    def from_pretrained(cls, model_id: str = "amazon/chronos-t5-small", device: str | None = None,
                        max_context: int = 512) -> "ChronosSampler":
        import torch
        from chronos import ChronosPipeline

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipe = ChronosPipeline.from_pretrained(model_id, device_map=dev, torch_dtype=torch.float32)
        return cls(pipe, model_id, max_context)

    def device(self) -> str:
        return str(getattr(getattr(self.pipe, "model", None), "device", "unknown"))

    def sample(self, ctx: pd.DataFrame, y_ts: pd.DatetimeIndex, n_paths: int, seed: int,
               T: float = 1.0, top_p: float = 0.9, batch: int = 64) -> dict[str, np.ndarray]:
        """Return {field: float32 [n_paths, len(y_ts)]}. Chronos' own sampling defaults are used;
        T / top_p are Kronos settings and are ignored here."""
        import torch

        _seed_all(seed)
        x = torch.tensor(ctx["close"].to_numpy(np.float32)[-self.max_context:])
        s = self.pipe.predict(x, prediction_length=len(y_ts), num_samples=n_paths)  # (1, n, h)
        close = np.asarray(s[0], dtype=np.float32)
        close = np.maximum(close, np.float32(1e-6) * np.float32(ctx["close"].iloc[-1]))  # keep log returns finite
        return {c: close for c in OHLCV}


def make_sampler(cfg: dict, kronos_path=None, device=None):
    """Pick the sampler from cfg['model_family'] (kronos | chronos)."""
    fam = cfg.get("model_family", "kronos")
    if fam == "chronos":
        return ChronosSampler.from_pretrained(cfg["model_id"], device, cfg["max_context"])
    from .sampler import KronosSampler

    return KronosSampler.from_pretrained(cfg["model_id"], cfg["tokenizer_id"], kronos_path, device,
                                         cfg["max_context"], cfg["amp"])
