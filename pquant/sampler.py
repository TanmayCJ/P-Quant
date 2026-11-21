"""Kronos path sampling.

KronosPredictor averages samples internally when sample_count > 1, so we get
individual trajectories by batching N copies of the same context with
sample_count=1. The vendored Kronos code is left unmodified.
"""
from __future__ import annotations

import contextlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLS, OHLCV

PATH_FIELDS = ["open", "high", "low", "close"]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class KronosSampler:
    def __init__(self, predictor, model_id: str = "custom", max_context: int = 512, amp: bool = False):
        self.predictor = predictor
        self.model_id = model_id
        self.max_context = max_context
        self.amp = amp

    @classmethod
    def from_pretrained(cls, model_id: str, tokenizer_id: str, kronos_path: str | Path | None = None,
                        device: str | None = None, max_context: int = 512,
                        amp: bool = False) -> "KronosSampler":
        if kronos_path and str(kronos_path) not in sys.path:
            sys.path.insert(0, str(kronos_path))
        from model import Kronos, KronosPredictor, KronosTokenizer  # vendored Kronos repo

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        return cls(predictor, model_id, max_context, amp)

    def device(self) -> str:
        return str(getattr(self.predictor, "device", "unknown"))

    def _autocast(self):
        if not self.amp:
            return contextlib.nullcontext()
        import torch

        dev = self.device()
        if not (dev.startswith("cuda") or (dev == "unknown" and torch.cuda.is_available())):
            raise RuntimeError("amp=True requires a CUDA device")
        return torch.autocast("cuda", dtype=torch.float16)

    def sample(self, ctx: pd.DataFrame, y_ts: pd.DatetimeIndex, n_paths: int, seed: int,
               T: float = 1.0, top_p: float = 0.9, batch: int = 64) -> dict[str, np.ndarray]:
        """Return {field: float32 array [n_paths, len(y_ts)]}."""
        if len(ctx) > self.max_context:
            raise ValueError(f"context {len(ctx)} > max_context {self.max_context}")
        x_ts = pd.Series(pd.DatetimeIndex(ctx.index))
        y = pd.Series(pd.DatetimeIndex(y_ts))
        x = ctx[COLS].reset_index(drop=True)
        out: dict[str, list] = {c: [] for c in OHLCV}
        done, chunk = 0, 0
        while done < n_paths:
            m = min(batch, n_paths - done)
            _seed_all(seed * 1000 + chunk)
            with self._autocast():
                dfs = self.predictor.predict_batch(
                    df_list=[x] * m, x_timestamp_list=[x_ts] * m, y_timestamp_list=[y] * m,
                    pred_len=len(y), T=T, top_p=top_p, sample_count=1, verbose=False,
                )
            for d in dfs:
                for c in OHLCV:
                    out[c].append(d[c].to_numpy(np.float32))
            done += m
            chunk += 1
        return {c: np.stack(v) for c, v in out.items()}
