"""The champion model: loading, input preparation and scoring.

`prepare()` applies exactly the same rules as training (see notebooks/05_champion.ipynb):
the same feature order, the same text codes, the same missing-value markers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .reasons import top_reasons

MODEL_FILE = "lightgbm_model.txt"
PREPROCESS_FILE = "preprocess.json"
CHAMPION_FILE = "champion.json"


@dataclass
class Decision:
    fraud_score: float
    suspicious: bool
    latency_ms: float


class Champion:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        missing = [f for f in (MODEL_FILE, PREPROCESS_FILE, CHAMPION_FILE) if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing in {model_dir}: {', '.join(missing)}")

        self.booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
        self.prep = json.loads((model_dir / PREPROCESS_FILE).read_text())
        self.info = json.loads((model_dir / CHAMPION_FILE).read_text())

        self.features: list[str] = self.prep["features"]
        self.text_cols = set(self.prep["text_cols"])
        self.codes: dict[str, dict[str, int]] = self.prep["codes"]
        self.threshold: float = float(self.info["threshold"])

        self._num_cols = [c for c in self.features if c not in self.text_cols]
        self._num_idx = [self.features.index(c) for c in self._num_cols]
        self._text_idx = {c: self.features.index(c) for c in self.text_cols}

        # Warm up so the first real request is as fast as the rest
        self.score({})

    def prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Raw payments -> model input matrix, using the training rules."""
        X = df.reindex(columns=self.features)
        out = np.empty((len(df), len(self.features)), dtype="float32")
        num = X[self._num_cols].apply(pd.to_numeric, errors="coerce") if len(df) > 1 else X[self._num_cols]
        num = num.to_numpy(dtype="float32", na_value=np.nan)
        out[:, self._num_idx] = np.where(np.isnan(num), self.prep["missing_value"], num)
        miss, unseen = self.prep["missing_text"], self.prep["unseen_text"]
        for c, j in self._text_idx.items():
            codes = self.codes[c]
            out[:, j] = [miss if (v is None or v != v) else codes.get(str(v), unseen) for v in X[c].tolist()]
        return out

    def vector(self, features: dict) -> np.ndarray:
        """One payment {column: value} -> one model input row, same rules as prepare()."""
        out = np.full((1, len(self.features)), self.prep["missing_value"], dtype="float32")
        miss, unseen = self.prep["missing_text"], self.prep["unseen_text"]
        for j, c in enumerate(self.features):
            v = features.get(c)
            if c in self.text_cols:
                out[0, j] = miss if (v is None or v != v) else self.codes[c].get(str(v), unseen)
            elif v is not None:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if f == f:  # not NaN
                    out[0, j] = f
        return out

    def score(self, features: dict) -> Decision:
        """Score one payment given as {column: value}."""
        t0 = time.perf_counter()
        p = float(self.booster.predict(self.vector(features), num_threads=1)[0])
        return Decision(fraud_score=p, suspicious=p >= self.threshold,
                        latency_ms=(time.perf_counter() - t0) * 1000)

    def explain(self, features: dict, suspicious: bool, k: int = 3) -> tuple[list[dict], float]:
        """Top k reasons for the decision (SHAP values computed by LightGBM), and the time taken."""
        t0 = time.perf_counter()
        contrib = self.booster.predict(self.vector(features), pred_contrib=True, num_threads=1)[0]
        reasons = top_reasons(contrib[:-1], self.features, features, suspicious, k)  # last = base value
        return reasons, (time.perf_counter() - t0) * 1000

    def summary(self) -> dict:
        return {
            "model": self.info.get("model"),
            "trees": self.booster.num_trees(),
            "features": len(self.features),
            "threshold": self.threshold,
            "threshold_policy": self.info.get("threshold_policy"),
            "replay": self.info.get("replay"),
        }

