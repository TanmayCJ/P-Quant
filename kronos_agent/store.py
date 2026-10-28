"""Append-only, resumable Parquet store (safe across Colab disconnects)."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pandas as pd


class ResultStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def files(self) -> list[Path]:
        return sorted(self.root.glob("part-*.parquet"))

    def done_keys(self) -> set[str]:
        keys: set[str] = set()
        for f in self.files():
            keys.update(pd.read_parquet(f, columns=["key"])["key"].tolist())
        return keys

    def append(self, records: list[dict]) -> Path | None:
        if not records:
            return None
        name = f"part-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = self.root / f".{name}.tmp"
        pd.DataFrame(records).to_parquet(tmp, index=False)
        os.replace(tmp, self.root / name)
        return self.root / name

    def load(self) -> pd.DataFrame:
        files = self.files()
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df.drop_duplicates("key", keep="last").sort_values("origin_ts").reset_index(drop=True)
