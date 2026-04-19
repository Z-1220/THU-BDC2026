"""Pointwise Stock Transformer，封装为 Qlib `Model` 接口。

设计要点：
1. Pointwise：对每只股票独立预测未来 5 日开盘收益率（回归，MSE），
   去掉旧版本中的 CrossStockAttention；
2. 输入：Qlib `TSDatasetH` 产出的 (N, seq_len, feat_dim) 序列 + label；
3. 兼容 Qlib workflow：实现 fit / predict，predict 返回带 (datetime, instrument)
   MultiIndex 的 pd.Series。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.weight import Reweighter
from qlib.log import get_module_logger
from qlib.model.base import Model


# ----------------------------------------------------------------------
# 网络模块
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """时序维度上的加权聚合（把 seq_len 池化成 1）。"""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D] -> [B, D]
        weights = self.attention(x)
        pooled = torch.sum(x * weights, dim=1)
        return self.dropout(pooled)


class PointwiseStockTransformer(nn.Module):
    """去掉 CrossStockAttention 的 pointwise 回归 Transformer。"""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_attention = FeatureAttention(d_model, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        proj = self.input_proj(x)
        proj = self.pos_encoder(proj)
        temporal = self.temporal_encoder(proj)
        pooled = self.feature_attention(temporal)
        return self.head(pooled).squeeze(-1)


# ----------------------------------------------------------------------
# Qlib Model 封装
# ----------------------------------------------------------------------
class PointwiseTransformerModel(Model):
    """Qlib 标准 Model 接口包装。"""

    def __init__(
        self,
        d_feat: int = 6,
        seq_len: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_size: int = 256,
        n_epochs: int = 50,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 5.0,
        enable_grad_clip: bool = True,
        early_stop: int = 10,
        loss: str = "mse",
        scheduler: str = "cosine",
        num_workers: int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.logger = get_module_logger("PointwiseTransformerModel")
        self.d_feat = d_feat
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.enable_grad_clip = enable_grad_clip
        self.early_stop = early_stop
        self.loss_name = loss
        self.scheduler_name = scheduler
        self.num_workers = num_workers
        self.seed = seed

        self._set_seed(seed)
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._net: PointwiseStockTransformer | None = None
        self._fitted = False

    # --------------------- 工具 ---------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_net(self, input_dim: int) -> None:
        self._net = PointwiseStockTransformer(
            input_dim=input_dim,
            seq_len=self.seq_len,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)
        self.logger.info(f"Model params: {count_parameters(self._net):,}")

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return torch.mean((pred - target) ** 2)
        if self.loss_name == "mae":
            return torch.mean(torch.abs(pred - target))
        raise ValueError(f"Unknown loss: {self.loss_name}")

    # --------------------- Qlib Model 接口 ---------------------
    def fit(
        self,
        dataset: DatasetH,
        evals_result: dict | None = None,
        reweighter: Reweighter | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        assert isinstance(dataset, TSDatasetH), (
            f"PointwiseTransformerModel 需要 TSDatasetH，收到 {type(dataset).__name__}"
        )

        train_ds = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
        valid_ds = dataset.prepare("valid", col_set=["feature", "label"], data_key="learn")

        # 推断特征维度：TSDataSampler 返回 ndarray，第一个样本形状 (seq_len, n_features+label_cols)
        sample0 = train_ds[0]
        # sample0 shape: (seq_len, n_features + n_labels)
        input_dim = sample0.shape[-1] - 1
        self.d_feat = input_dim
        self._build_net(input_dim)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epochs
            )
        else:
            scheduler = None

        if evals_result is None:
            evals_result = {}
        evals_result.setdefault("train", [])
        evals_result.setdefault("valid", [])

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stop_count = 0

        for epoch in range(self.n_epochs):
            train_loss = self._run_epoch(train_loader, optimizer, training=True)
            valid_loss = self._run_epoch(valid_loader, None, training=False)

            evals_result["train"].append(train_loss)
            evals_result["valid"].append(valid_loss)
            self.logger.info(
                f"Epoch {epoch+1}/{self.n_epochs} | train {train_loss:.6f} | valid {valid_loss:.6f}"
            )

            if scheduler is not None:
                scheduler.step()

            if valid_loss < best_val - 1e-6:
                best_val = valid_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
                stop_count = 0
            else:
                stop_count += 1
                if stop_count >= self.early_stop:
                    self.logger.info(f"Early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._fitted = True

        if save_path:
            torch.save(self._net.state_dict(), save_path)

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        training: bool,
    ) -> float:
        assert self._net is not None
        self._net.train(training)
        total = 0.0
        count = 0
        for batch in loader:
            # batch: ndarray (B, seq_len, n_features+1) -> tensor
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch)
            batch = batch.float().to(self.device)
            feat = batch[..., :-1]
            target = batch[:, -1, -1]  # 最后一天对应的 label

            if training:
                optimizer.zero_grad()
            pred = self._net(feat)
            mask = ~torch.isnan(target)
            if mask.sum() == 0:
                continue
            loss = self._loss(pred[mask], target[mask])
            if training:
                loss.backward()
                if self.enable_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.max_grad_norm)
                optimizer.step()

            total += loss.item() * mask.sum().item()
            count += int(mask.sum().item())
        return total / max(count, 1)

    def predict(
        self,
        dataset: DatasetH,
        segment: str = "test",
        **kwargs: Any,
    ) -> pd.Series:
        if not self._fitted or self._net is None:
            raise RuntimeError("模型未训练，无法调用 predict()")

        assert isinstance(dataset, TSDatasetH)
        test_ds = dataset.prepare(segment, col_set=["feature", "label"], data_key="infer")
        loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._net.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch)
                batch = batch.float().to(self.device)
                feat = batch[..., :-1]
                out = self._net(feat).detach().cpu().numpy()
                preds.append(out)

        pred_arr = np.concatenate(preds, axis=0)
        # TSDataSampler 保留原始 (datetime, instrument) MultiIndex
        index = test_ds.get_index()
        return pd.Series(pred_arr, index=index, name="score")
