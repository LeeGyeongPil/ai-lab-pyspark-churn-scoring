"""Where each step reads and writes its files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_CUTOFF = date(2026, 6, 1)
OBS_DAYS = 120  # observation window: days before the cutoff used for features
LABEL_DAYS = 30  # label window: days after the cutoff used to decide churn


@dataclass(frozen=True)
class Layout:
    data_dir: Path
    model_dir: Path
    out_dir: Path

    @property
    def customers(self) -> str:
        return str(self.data_dir / "customers.parquet")

    @property
    def events(self) -> str:
        return str(self.data_dir / "events.parquet")

    @property
    def features(self) -> str:
        return str(self.data_dir / "features.parquet")

    @property
    def meta(self) -> Path:
        return self.data_dir / "meta.json"

    @property
    def threshold_file(self) -> Path:
        return self.model_dir / "threshold.json"

    @property
    def targets_csv(self) -> Path:
        return self.out_dir / "targets.csv"

    def model_path(self, name: str) -> str:
        return str(self.model_dir / name)

    def read_meta(self) -> dict:
        if self.meta.exists():
            return json.loads(self.meta.read_text(encoding="utf-8"))
        return {}

    def write_meta(self, meta: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
