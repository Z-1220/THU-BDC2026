"""Kronos Qlib Model wrapper — zero-shot inference with pretrained weights.

Kronos is a decoder-only Transformer pretrained on OHLCV data from 45+ global
exchanges. This wrapper loads pretrained weights locally and uses
KronosPredictor to forecast future OHLCV, then converts predictions into
scalar scores for stock ranking.

fit() is a no-op (pretrained weights are frozen). predict() bypasses
TSDatasetH and works directly with raw OHLCV from stock_data.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from qlib.data.dataset import DatasetH
from qlib.log import get_module_logger
from qlib.model.base import Model

from .kronos_src import Kronos, KronosPredictor, KronosTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Column mapping: stock_data.csv -> Kronos expected names
_COL_MAP = {
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def _make_timestamps(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build Kronos-compatible time features from a DatetimeIndex (daily freq)."""
    return pd.DataFrame(
        {
            "minute": 0,
            "hour": 0,
            "weekday": dates.weekday,
            "day": dates.day,
            "month": dates.month,
        },
        index=dates,
    )


class KronosModel(Model):
    """Qlib Model wrapping pretrained Kronos for zero-shot stock scoring."""

    def __init__(
        self,
        model_name: str = "small",
        pretrained_dir: str = "./model/kronos_pretrained",
        finetuned_dir: str | None = None,
        max_context: int = 512,
        pred_len: int = 5,
        device: str | None = None,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: "small" or "base".
            pretrained_dir: local directory containing Kronos-Tokenizer-base/,
                            Kronos-small/, Kronos-base/.
            finetuned_dir: optional directory with fine-tuned model weights.
                           If provided, loads Kronos from here instead of
                           pretrained_dir. Tokenizer still from pretrained_dir.
            max_context: max historical tokens for Kronos (512 for small/base).
            pred_len: number of future days to predict (default 5 for 1-week).
            device: torch device string (auto-detect if None).
            T: sampling temperature.
            top_p: nucleus sampling threshold.
            sample_count: number of Monte Carlo samples (averaged).
            seed: random seed.
        """
        self.logger = get_module_logger("KronosModel")
        self.model_name = model_name
        self.pretrained_dir = Path(pretrained_dir)
        self.finetuned_dir = Path(finetuned_dir) if finetuned_dir else None
        self.max_context = max_context
        self.pred_len = pred_len
        self.T = T
        self.top_p = top_p
        self.sample_count = sample_count
        self.seed = seed

        self._set_seed(seed)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._fitted = False
        self._predictor: KronosPredictor | None = None
        self._load_model()

        # Pre-load raw OHLCV data for fast access in predict()
        self._ohlcv_df = self._load_ohlcv_data()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        tokenizer_dir = self.pretrained_dir / "Kronos-Tokenizer-base"
        if self.finetuned_dir is not None:
            model_dir = self.finetuned_dir
            self.logger.info(f"Loading fine-tuned Kronos-{self.model_name} from {model_dir}")
        else:
            model_dir = self.pretrained_dir / f"Kronos-{self.model_name}"
            self.logger.info(f"Loading Kronos-{self.model_name} from {model_dir}")

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Kronos model not found at {model_dir}. "
                f"Run: python scripts/download_kronos_models.py"
            )
        if not tokenizer_dir.exists():
            raise FileNotFoundError(
                f"Kronos tokenizer not found at {tokenizer_dir}. "
                f"Run: python scripts/download_kronos_models.py"
            )

        self.logger.info(f"Loading Kronos-{self.model_name} from {model_dir}")
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        model = Kronos.from_pretrained(str(model_dir), local_files_only=True)
        self._predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.max_context
        )
        self.logger.info("Kronos model loaded.")

    @staticmethod
    def _load_ohlcv_data() -> pd.DataFrame:
        csv_path = _PROJECT_ROOT / "data" / "stock_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"stock_data.csv not found at {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["日期"])
        df = df.rename(columns={**{k: v for k, v in _COL_MAP.items()}, "股票代码": "code"})
        # Kronos expects code as string
        df["code"] = df["code"].astype(str)
        return df

    # ------------------------------------------------------------------
    # Qlib Model interface
    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: DatasetH,
        evals_result: dict | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """No-op: Kronos is used with frozen pretrained weights."""
        self.logger.info(
            "KronosModel.fit() is a no-op — using frozen pretrained weights. "
            "The model produces zero-shot predictions."
        )
        self._fitted = True

    def predict(
        self,
        dataset: DatasetH,
        segment: str = "test",
        **kwargs: Any,
    ) -> pd.Series:
        """Run Kronos inference and return scalar scores for ranking.

        Returns pd.Series with (datetime, instrument) MultiIndex.
        """
        if not self._fitted or self._predictor is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get the index we need to predict for (Friday signal dates only,
        # after FridayFilterProcessor).
        # DatasetH.prepare returns DataFrame (.index), TSDatasetH returns
        # TSDataSampler (.get_index()). Handle both.
        test_ds = dataset.prepare(segment, col_set=["feature", "label"], data_key="infer")
        if hasattr(test_ds, "get_index"):
            index = test_ds.get_index()
        else:
            index = test_ds.index

        if len(index) == 0:
            self.logger.warning("Empty test index — returning empty Series.")
            return pd.Series([], index=index, name="score", dtype=float)

        signal_dates = sorted(index.get_level_values(0).unique())
        self.logger.info(
            f"Kronos predict: {len(signal_dates)} signal dates, "
            f"{len(index)} (date, instrument) pairs"
        )

        all_parts: list[pd.Series] = []
        for dt in signal_dates:
            instruments = list(
                index.get_level_values(1)[index.get_level_values(0) == dt]
            )
            part = self._predict_signal_date(dt, instruments)
            all_parts.append(part)

        result = pd.concat(all_parts)
        result.name = "score"
        return result

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    @staticmethod
    def _csv_code(qlib_instrument: str) -> str:
        """Convert Qlib instrument code (SH600000) to CSV code (600000)."""
        if qlib_instrument.startswith(("SH", "SZ")):
            return qlib_instrument[2:]
        return qlib_instrument

    def _predict_signal_date(
        self, signal_date: pd.Timestamp, instruments: list[str]
    ) -> pd.Series:
        """Predict scores for all instruments at a single signal date."""
        df = self._ohlcv_df
        required_cols = ["open", "high", "low", "close", "volume", "amount"]

        # Filter data up to signal_date (exclusive — predict future)
        hist = df[df["日期"] <= signal_date].copy()

        future_df = df[df["日期"] > signal_date].copy()
        future_dates_all = sorted(future_df["日期"].unique())
        future_dates = future_dates_all[: self.pred_len]

        if len(future_dates) < self.pred_len:
            self.logger.warning(
                f"Only {len(future_dates)} future dates available at {signal_date.date()}, "
                f"expected {self.pred_len}. Using available dates."
            )
            if len(future_dates) == 0:
                return pd.Series(
                    [0.0] * len(instruments),
                    index=pd.MultiIndex.from_tuples(
                        [(signal_date, inst) for inst in instruments],
                        names=["datetime", "instrument"],
                    ),
                    name="score",
                )
            # Use fewer prediction days if data is still sufficient
            actual_pred_len = len(future_dates)
        else:
            actual_pred_len = self.pred_len

        x_dfs: list[pd.DataFrame] = []
        x_ts_list: list[pd.Series] = []
        valid_instruments: list[str] = []

        y_timestamps = pd.DatetimeIndex(future_dates[:actual_pred_len])

        for inst in instruments:
            csv_code = self._csv_code(inst)
            inst_hist = hist[hist["code"] == csv_code].sort_values("日期")
            if len(inst_hist) < 60:  # need reasonable history
                continue

            # Take last max_context days
            inst_hist = inst_hist.tail(self.max_context)
            x_df = inst_hist[required_cols].ffill().bfill()
            x_dfs.append(x_df)
            x_ts_list.append(inst_hist["日期"])
            valid_instruments.append(inst)

        if not valid_instruments:
            return pd.Series(
                [], index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]), name="score", dtype=float
            )

        # Use individual predict() to avoid equal-length constraint of predict_batch
        pred_dfs = []
        for i in range(len(x_dfs)):
            try:
                pdf = self._predictor.predict(
                    df=x_dfs[i],
                    x_timestamp=x_ts_list[i],
                    y_timestamp=pd.Series(y_timestamps),
                    pred_len=actual_pred_len,
                    T=self.T,
                    top_k=0,
                    top_p=self.top_p,
                    sample_count=self.sample_count,
                    verbose=False,
                )
                pred_dfs.append(pdf)
            except Exception:
                pred_dfs.append(None)

        # Compute scores: predicted close return over the forecast horizon
        scores = []
        for i, inst in enumerate(valid_instruments):
            if pred_dfs[i] is None:
                scores.append(0.0)
                continue
            last_close = x_dfs[i]["close"].iloc[-1]
            pred_close = pred_dfs[i]["close"].iloc[-1]
            score = (pred_close - last_close) / (last_close + 1e-12)
            scores.append(score)

        # Fill zero for instruments we couldn't process
        final_scores = []
        inst_set = set(valid_instruments)
        for inst in instruments:
            if inst in inst_set:
                idx = valid_instruments.index(inst)
                final_scores.append(scores[idx])
            else:
                final_scores.append(0.0)

        return pd.Series(
            final_scores,
            index=pd.MultiIndex.from_tuples(
                [(signal_date, inst) for inst in instruments],
                names=["datetime", "instrument"],
            ),
            name="score",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        import os
        import random

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Don't pickle the predictor or cached DataFrame
        state["_predictor"] = None
        state["_ohlcv_df"] = None
        state.pop("device", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._load_model()
        self._ohlcv_df = self._load_ohlcv_data()
