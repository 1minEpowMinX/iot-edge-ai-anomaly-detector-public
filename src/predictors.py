"""Predictor abstraction unifying torch and non-torch models behind one API."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .losses import build_loss


class BasePredictor(ABC):
    """Common interface for next-step predictors."""

    name: str = "base"
    _train_time_s: float = 0.0

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, list[float]]:
        """Train and return history dict containing at least 'train', 'val', 'lr'."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, window, n_features) -> (batch, n_features)."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters."""
        ...

    @property
    def train_time_s(self) -> float:
        """Wall-clock training time in seconds for the last ``fit()`` call."""
        return self._train_time_s

    def measure_inference_ms(
        self,
        X_sample: np.ndarray,
        n_repeat: int = 50,
        n_warmup: int = 5,
    ) -> float:
        """Median single-window inference latency in milliseconds.

        Median (not mean) so a single GPU-contention spike doesn't dominate.
        Warmup absorbs CUDA kernel compilation / cuDNN autotuning.
        """
        sample = X_sample[:1]
        for _ in range(n_warmup):
            self.predict(sample)
        times = np.empty(n_repeat, dtype=np.float64)
        for i in range(n_repeat):
            t0 = time.perf_counter()
            self.predict(sample)
            times[i] = time.perf_counter() - t0
        return float(np.median(times)) * 1000.0


class TorchPredictor(BasePredictor):
    """Wraps an ``nn.Module`` with an AdamW + ReduceLROnPlateau training loop.

    Loss function and gradient clipping come from ``TrainConfig``.
    """

    def __init__(
        self,
        net: nn.Module,
        name: str,
        config: TrainConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        """Initialise the predictor.

        Args:
            net: The ``nn.Module`` to train and run.
            name: Human-readable predictor name.
            config: Training configuration; defaults to ``TrainConfig()``.
            device: Torch device string (``"cpu"`` or ``"cuda"``).
        """
        self.net = net.to(device)
        self.name = name
        self.config = config or TrainConfig()
        self.device = device
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None
        self.stopped_epoch: int | None = None  # set if early-stopped

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, list[float]]:
        """Train the network with early stopping and LR scheduling.

        Args:
            X_train: Training input windows.
            y_train: Training next-step targets.
            X_val: Validation input windows.
            y_val: Validation next-step targets.

        Returns:
            History dict with ``"train"``, ``"val"`` and ``"lr"`` lists.
        """
        cfg = self.config
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
        criterion = build_loss(cfg.loss, cfg.huber_delta)
        history: dict[str, list[float]] = {"train": [], "val": [], "lr": []}

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        patience_counter = 0
        # Tracks whether we've seen an improvement since the previous lr drop.
        # Used to disable patience-reset on lr drops that follow a fruitless interval.
        improved_since_last_lr_drop = True
        self.stopped_epoch = None

        if cfg.verbose:
            es_info = (
                f"  early_stop={cfg.early_stopping_patience}"
                if cfg.early_stopping
                else ""
            )
            print(
                f"[{self.name}] параметрів: {self.n_params:,}  "
                f"loss={cfg.loss}  grad_clip={cfg.grad_clip}{es_info}"
            )

        t0 = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimizer)
            val_loss = self._run_eval(val_loader, criterion)

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            lr_dropped = new_lr < prev_lr

            history["train"].append(train_loss)
            history["val"].append(val_loss)
            history["lr"].append(new_lr)

            # track best — meaningful improvement only (filters noise)
            improved = val_loss < best_val - cfg.early_stopping_min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                improved_since_last_lr_drop = True
                if cfg.restore_best_weights:
                    best_state = {
                        k: v.detach().clone() for k, v in self.net.state_dict().items()
                    }
            else:
                patience_counter += 1

            # LR drop changes training dynamics — give the new lr a fair window
            # before declaring "no improvement". But only do this if the
            # previous lr level was actually productive (it improved val at
            # least once). If nothing improved at the higher lr, lowering it
            # further is unlikely to help — let ES fire normally.
            reset_now = (
                lr_dropped
                and cfg.early_stopping_reset_on_lr_drop
                and (
                    improved_since_last_lr_drop
                    or not cfg.early_stopping_require_progress_for_reset
                )
            )
            if reset_now:
                patience_counter = 0
            if lr_dropped:
                # arm the flag for the next lr-drop interval regardless
                improved_since_last_lr_drop = False

            if cfg.verbose:
                if lr_dropped:
                    if reset_now:
                        note = " (patience reset)"
                    elif cfg.early_stopping_reset_on_lr_drop:
                        note = " (no reset — last lr drop was unproductive)"
                    else:
                        note = ""
                    print(
                        f"[{self.name}][epoch {epoch:>3}/{cfg.epochs}] "
                        f"lr {prev_lr:.2e} -> {new_lr:.2e}{note}"
                    )
                if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
                    print(
                        f"[{self.name}][epoch {epoch:>3}/{cfg.epochs}] "
                        f"train={train_loss:.5f}  val={val_loss:.5f}  "
                        f"lr={new_lr:.2e}"
                    )

            if cfg.early_stopping and patience_counter >= cfg.early_stopping_patience:
                self.stopped_epoch = epoch
                if cfg.verbose:
                    print(
                        f"[{self.name}] ! early stop на епосі {epoch} "
                        f"(найкращий val={best_val:.5f} @ епоха {best_epoch})"
                    )
                break

        self._train_time_s = time.perf_counter() - t0
        self.best_epoch = best_epoch
        self.best_val_loss = best_val if best_epoch > 0 else None

        # restore the best snapshot — return the best model, not the last
        if (
            cfg.restore_best_weights
            and best_state is not None
            and best_epoch != len(history["val"])
        ):
            self.net.load_state_dict(best_state)
            if cfg.verbose:
                print(
                    f"[{self.name}] << ваги відновлено з епохи {best_epoch} "
                    f"(val={best_val:.5f})"
                )

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass and return next-step predictions as a NumPy array.

        Args:
            X: Input windows, shape ``(batch, window, n_features)``.

        Returns:
            Predictions, shape ``(batch, n_features)``.
        """
        self.net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(self.device)
            return self.net(x_t).cpu().numpy()

    @property
    def n_params(self) -> int:
        """Number of trainable parameters in the wrapped network."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def state_dict(self) -> dict[str, Any]:
        """Return the wrapped network's ``state_dict``."""
        return self.net.state_dict()

    def _run_epoch(self, loader, criterion, optimizer) -> float:
        """Run one training epoch and return the mean training loss."""
        self.net.train()
        running = 0.0
        clip = self.config.grad_clip
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.net(xb), yb)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
            optimizer.step()
            running += loss.item() * xb.size(0)
        return running / len(loader.dataset)

    def _run_eval(self, loader, criterion) -> float:
        """Evaluate on a loader and return the mean loss (no gradients)."""
        self.net.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                running += criterion(self.net(xb), yb).item() * xb.size(0)
        return running / len(loader.dataset)


class MovingAveragePredictor(BasePredictor):
    """Naive baseline: prediction = mean over the input window."""

    def __init__(self, name: str = "MovingAvg") -> None:
        """Initialise the baseline.

        Args:
            name: Human-readable predictor name.
        """
        self.name = name

    def fit(self, X_train, y_train, X_val, y_val) -> dict[str, list[float]]:
        """Return an empty history — the baseline requires no training."""
        return {"train": [], "val": [], "lr": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the next step as the mean over each input window."""
        return X.mean(axis=1).astype(np.float32)

    @property
    def n_params(self) -> int:
        """Number of trainable parameters (always 0 for this baseline)."""
        return 0
