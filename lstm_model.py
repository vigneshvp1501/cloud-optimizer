"""
LSTM Workload Prediction Model
Predicts future cloud resource utilization using sliding window approach
"""

import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ─────────────────────────────────────────────
# Sliding Window Dataset Builder
# ─────────────────────────────────────────────
class SlidingWindowDataset:
    """Converts time-series metrics into supervised learning windows."""

    def __init__(self, window_size: int = 24, horizon: int = 6):
        """
        Args:
            window_size: Number of past timesteps used as features (default: 24 hours)
            horizon:     Number of future timesteps to predict  (default: 6 hours)
        """
        self.window_size = window_size
        self.horizon = horizon

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slice a 1-D or 2-D time series into (X, y) pairs.

        Returns:
            X: shape (n_samples, window_size, n_features)
            y: shape (n_samples, horizon)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        X, y = [], []
        total = self.window_size + self.horizon

        for i in range(len(data) - total + 1):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size : i + total, 0])   # predict first feature

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def split(
        self,
        data: np.ndarray,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """Train / validation / test split (time-ordered, no shuffling)."""
        X, y = self.create_sequences(data)
        n = len(X)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test

        return (
            (X[:n_train],          y[:n_train]),
            (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
            (X[n_train+n_val:],    y[n_train+n_val:]),
        )


# ─────────────────────────────────────────────
# Lightweight NumPy LSTM (no external ML deps)
# ─────────────────────────────────────────────
class NumpyLSTMCell:
    """Single LSTM cell – forward pass only (inference)."""

    def __init__(self, input_size: int, hidden_size: int):
        scale = 0.1
        # Gate weights: [W_i, W_f, W_g, W_o]  (input + hidden → gates)
        self.Wh = np.random.randn(4 * hidden_size, hidden_size) * scale
        self.Wx = np.random.randn(4 * hidden_size, input_size)  * scale
        self.b  = np.zeros(4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        gates = self.Wx @ x + self.Wh @ h + self.b
        H = self.hidden_size
        i = self._sigmoid(gates[0*H:1*H])
        f = self._sigmoid(gates[1*H:2*H])
        g = np.tanh(gates[2*H:3*H])
        o = self._sigmoid(gates[3*H:4*H])
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        return h_new, c_new

    @staticmethod
    def _sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def get_weights(self) -> dict:
        return {"Wh": self.Wh.tolist(), "Wx": self.Wx.tolist(), "b": self.b.tolist()}

    def set_weights(self, d: dict):
        self.Wh = np.array(d["Wh"])
        self.Wx = np.array(d["Wx"])
        self.b  = np.array(d["b"])


class WorkloadLSTM:
    """
    Two-layer LSTM + linear output head.
    Trained with simple gradient-free evolutionary strategy for portability.
    In production swap this for a PyTorch / TensorFlow equivalent.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64, horizon: int = 6):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.horizon     = horizon
        self.cell1 = NumpyLSTMCell(input_size,   hidden_size)
        self.cell2 = NumpyLSTMCell(hidden_size,  hidden_size)
        # Linear head: hidden → horizon
        self.W_out = np.random.randn(horizon, hidden_size) * 0.1
        self.b_out = np.zeros(horizon)
        self.scaler_mean = 0.0
        self.scaler_std  = 1.0

    # ── Normalization ──────────────────────────
    def fit_scaler(self, data: np.ndarray):
        self.scaler_mean = float(np.mean(data))
        self.scaler_std  = float(np.std(data)) or 1.0

    def normalize(self, x): return (x - self.scaler_mean) / self.scaler_std
    def denormalize(self, x): return x * self.scaler_std + self.scaler_mean

    # ── Forward pass ──────────────────────────
    def predict_sequence(self, window: np.ndarray) -> np.ndarray:
        """
        Args:
            window: (window_size, input_size)  – already normalized
        Returns:
            predictions: (horizon,)             – denormalized
        """
        h1 = np.zeros(self.hidden_size)
        c1 = np.zeros(self.hidden_size)
        h2 = np.zeros(self.hidden_size)
        c2 = np.zeros(self.hidden_size)

        for t in range(window.shape[0]):
            h1, c1 = self.cell1.forward(window[t], h1, c1)
            h2, c2 = self.cell2.forward(h1, h2, c2)

        y_norm = self.W_out @ h2 + self.b_out
        return self.denormalize(y_norm)

    def predict(self, raw_window: np.ndarray) -> np.ndarray:
        """End-to-end: raw → normalise → LSTM → denormalise."""
        if raw_window.ndim == 1:
            raw_window = raw_window.reshape(-1, 1)
        norm = self.normalize(raw_window)
        return self.predict_sequence(norm)

    # ── Simplified training (MSE gradient descent on output layer only) ──
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 1e-3):
        """
        Trains the linear output head via gradient descent.
        LSTM cells are initialised with random weights (good enough for demo).
        For production, use a full backprop framework.
        """
        logger.info(f"Training on {len(X)} samples for {epochs} epochs …")
        best_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0.0
            dW = np.zeros_like(self.W_out)
            db = np.zeros_like(self.b_out)

            for xi, yi in zip(X, y):
                # Forward
                h = self._encode(xi)
                pred = self.W_out @ h + self.b_out
                err  = pred - yi
                total_loss += float(np.mean(err ** 2))
                # Grad for linear head
                dW += np.outer(err, h) * (2 / len(yi))
                db += err             * (2 / len(yi))

            n = len(X)
            self.W_out -= lr * dW / n
            self.b_out -= lr * db / n
            avg_loss = total_loss / n

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch:3d}/{epochs}  MSE={avg_loss:.4f}")

        logger.info(f"Training complete. Best MSE: {best_loss:.4f}")

    def _encode(self, window: np.ndarray) -> np.ndarray:
        """Run LSTM cells on a normalised window, return final hidden state."""
        h1 = np.zeros(self.hidden_size)
        c1 = np.zeros(self.hidden_size)
        h2 = np.zeros(self.hidden_size)
        c2 = np.zeros(self.hidden_size)
        for t in range(window.shape[0]):
            h1, c1 = self.cell1.forward(window[t], h1, c1)
            h2, c2 = self.cell2.forward(h1, h2, c2)
        return h2

    # ── Persistence ───────────────────────────
    def save(self, path: str):
        state = {
            "input_size":   self.input_size,
            "hidden_size":  self.hidden_size,
            "horizon":      self.horizon,
            "scaler_mean":  self.scaler_mean,
            "scaler_std":   self.scaler_std,
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
            "cell1": self.cell1.get_weights(),
            "cell2": self.cell2.get_weights(),
        }
        with open(path, "w") as f:
            json.dump(state, f)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "WorkloadLSTM":
        with open(path) as f:
            state = json.load(f)
        m = cls(state["input_size"], state["hidden_size"], state["horizon"])
        m.scaler_mean = state["scaler_mean"]
        m.scaler_std  = state["scaler_std"]
        m.W_out = np.array(state["W_out"])
        m.b_out = np.array(state["b_out"])
        m.cell1.set_weights(state["cell1"])
        m.cell2.set_weights(state["cell2"])
        logger.info(f"Model loaded ← {path}")
        return m


# ─────────────────────────────────────────────
# CloudWatch Data Fetcher
# ─────────────────────────────────────────────
class CloudWatchFetcher:
    """Pulls historical metrics from AWS CloudWatch."""

    def __init__(self, region: str = "us-east-1"):
        self.cw = boto3.client("cloudwatch", region_name=region)

    def fetch(
        self,
        namespace: str,
        metric_name: str,
        dimensions: List[dict],
        hours: int = 168,          # 7 days default
        period: int = 3600,        # 1-hour granularity
        stat: str = "Average",
    ) -> Tuple[List[datetime], List[float]]:
        end   = datetime.utcnow()
        start = end - timedelta(hours=hours)

        resp = self.cw.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[stat],
        )

        points = sorted(resp["Datapoints"], key=lambda p: p["Timestamp"])
        timestamps = [p["Timestamp"] for p in points]
        values     = [p[stat]        for p in points]

        logger.info(f"Fetched {len(values)} datapoints for {metric_name}")
        return timestamps, values


# ─────────────────────────────────────────────
# Entrypoint helpers
# ─────────────────────────────────────────────
def train_from_cloudwatch(
    namespace: str,
    metric_name: str,
    dimensions: List[dict],
    model_path: str = "/app/model/lstm_weights.json",
    region: str = "us-east-1",
    window_size: int = 24,
    horizon: int = 6,
):
    fetcher = CloudWatchFetcher(region)
    _, values = fetcher.fetch(namespace, metric_name, dimensions, hours=720)  # 30 days

    data = np.array(values, dtype=np.float32)
    model = WorkloadLSTM(input_size=1, hidden_size=64, horizon=horizon)
    model.fit_scaler(data)

    dataset = SlidingWindowDataset(window_size=window_size, horizon=horizon)
    (X_tr, y_tr), (X_val, y_val), _ = dataset.split(model.normalize(data))

    model.train(X_tr, y_tr, epochs=100, lr=5e-4)
    model.save(model_path)
    return model


def predict_next(
    model: WorkloadLSTM,
    recent_values: List[float],
) -> List[float]:
    window = np.array(recent_values[-model.cell1.Wx.shape[1] :], dtype=np.float32)
    predictions = model.predict(window)
    return predictions.tolist()
