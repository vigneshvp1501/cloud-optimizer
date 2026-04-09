"""
Tests for Cloud Resource Optimization System
Run: python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.lstm_model import (
    WorkloadLSTM,
    SlidingWindowDataset,
    NumpyLSTMCell,
)
from scaler.asg_controller import ASGController, ScalingPolicy


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def model():
    m = WorkloadLSTM(input_size=1, hidden_size=16, horizon=3)
    data = np.linspace(20, 80, 100, dtype=np.float32)
    m.fit_scaler(data)
    return m


@pytest.fixture
def policy():
    return ScalingPolicy(
        asg_name="test-asg",
        min_capacity=1,
        max_capacity=10,
        desired_capacity=2,
        scale_out_threshold=70.0,
        scale_in_threshold=30.0,
        scale_out_step=2,
        scale_in_step=1,
    )


@pytest.fixture
def controller():
    with tempfile.TemporaryDirectory() as tmp:
        yield ASGController(region="us-east-1", log_path=tmp)


# ─────────────────────────────────────────────
# LSTM Cell Tests
# ─────────────────────────────────────────────
class TestNumpyLSTMCell:
    def test_forward_shapes(self):
        cell = NumpyLSTMCell(input_size=4, hidden_size=8)
        x = np.random.randn(4)
        h = np.zeros(8)
        c = np.zeros(8)
        h_new, c_new = cell.forward(x, h, c)
        assert h_new.shape == (8,)
        assert c_new.shape == (8,)

    def test_forward_bounded(self):
        """tanh output of LSTM hidden state should be ∈ (-1, 1)."""
        cell = NumpyLSTMCell(input_size=2, hidden_size=4)
        x = np.ones(2) * 100          # extreme input
        h, c = cell.forward(x, np.zeros(4), np.zeros(4))
        assert np.all(np.abs(h) <= 1.0 + 1e-6)

    def test_weight_serialisation(self):
        cell = NumpyLSTMCell(input_size=3, hidden_size=5)
        w = cell.get_weights()
        cell2 = NumpyLSTMCell(input_size=3, hidden_size=5)
        cell2.set_weights(w)
        np.testing.assert_array_equal(cell.Wh, cell2.Wh)


# ─────────────────────────────────────────────
# Sliding Window Dataset Tests
# ─────────────────────────────────────────────
class TestSlidingWindowDataset:
    def test_sequence_count(self):
        ds   = SlidingWindowDataset(window_size=5, horizon=2)
        data = np.arange(20, dtype=np.float32)
        X, y = ds.create_sequences(data)
        expected = len(data) - 5 - 2 + 1   # = 14
        assert len(X) == expected == len(y)

    def test_shapes(self):
        ds = SlidingWindowDataset(window_size=6, horizon=3)
        X, y = ds.create_sequences(np.random.randn(50).astype(np.float32))
        assert X.shape[1:] == (6, 1)
        assert y.shape[1]  == 3

    def test_split_sizes(self):
        ds   = SlidingWindowDataset(window_size=4, horizon=2)
        data = np.arange(100, dtype=np.float32)
        (X_tr, _), (X_val, _), (X_te, _) = ds.split(data, val_ratio=0.1, test_ratio=0.1)
        assert len(X_tr) > len(X_val)
        assert len(X_tr) > len(X_te)


# ─────────────────────────────────────────────
# WorkloadLSTM Tests
# ─────────────────────────────────────────────
class TestWorkloadLSTM:
    def test_predict_shape(self, model):
        window = np.random.randn(24).astype(np.float32)
        preds  = model.predict(window)
        assert preds.shape == (3,)        # horizon = 3

    def test_predict_range(self, model):
        """Predictions should be within a reasonable range given scaler."""
        window = np.ones(24, dtype=np.float32) * 50.0
        preds  = model.predict(window)
        assert not np.any(np.isnan(preds))
        assert not np.any(np.isinf(preds))

    def test_save_load_roundtrip(self, model):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = WorkloadLSTM.load(path)
            window = np.random.randn(24).astype(np.float32)
            np.testing.assert_allclose(model.predict(window), loaded.predict(window), rtol=1e-5)
        finally:
            os.unlink(path)

    def test_normalise_denormalise(self, model):
        x = np.array([50.0], dtype=np.float32)
        assert abs(model.denormalize(model.normalize(x)) - x) < 1e-4

    def test_training_reduces_loss(self):
        """A few epochs of training should not blow up the model."""
        m = WorkloadLSTM(input_size=1, hidden_size=8, horizon=2)
        data = (np.sin(np.linspace(0, 4 * np.pi, 200)) * 20 + 50).astype(np.float32)
        m.fit_scaler(data)
        ds = SlidingWindowDataset(window_size=12, horizon=2)
        (X_tr, y_tr), _, _ = ds.split(m.normalize(data))
        # Should not raise
        m.train(X_tr, y_tr, epochs=5, lr=1e-3)
        preds = m.predict(data[-12:])
        assert preds.shape == (2,)


# ─────────────────────────────────────────────
# Scaling Policy & Decision Tests
# ─────────────────────────────────────────────
class TestScalingDecision:
    def test_scale_out_decision(self, controller, policy):
        assert controller.decide(policy, predicted_cpu=75.0) == "SCALE_OUT"

    def test_scale_in_decision(self, controller, policy):
        assert controller.decide(policy, predicted_cpu=20.0) == "SCALE_IN"

    def test_no_action_decision(self, controller, policy):
        assert controller.decide(policy, predicted_cpu=50.0) == "NO_ACTION"

    def test_boundary_scale_out(self, controller, policy):
        assert controller.decide(policy, predicted_cpu=70.0) == "SCALE_OUT"

    def test_boundary_scale_in(self, controller, policy):
        assert controller.decide(policy, predicted_cpu=30.0) == "SCALE_IN"

    def test_cooldown_blocks_action(self, controller, policy):
        """Set last-scale timestamp to now, next scale should be blocked."""
        from datetime import datetime
        controller._last_scale[policy.asg_name] = datetime.utcnow()
        assert controller._in_cooldown(policy.asg_name, policy.cooldown_seconds)

    def test_log_event_written(self, controller, policy):
        """Scaling events should be persisted to disk."""
        from scaler.asg_controller import ScalingEvent
        event = ScalingEvent(
            timestamp="2024-01-01T00:00:00",
            asg_name=policy.asg_name,
            action="NO_ACTION",
            reason="test",
            old_capacity=2,
            new_capacity=2,
            predicted_load=50.0,
        )
        controller._log_event(event)
        history = controller.load_history(policy.asg_name)
        assert len(history) >= 1
        assert history[-1].action == "NO_ACTION"


# ─────────────────────────────────────────────
# End-to-end smoke test (no AWS calls)
# ─────────────────────────────────────────────
class TestEndToEnd:
    def test_full_pipeline_synthetic(self):
        """
        Simulate a complete tick: generate data → train → predict → scaling decision.
        No AWS calls required.
        """
        # Generate synthetic workload
        t    = np.linspace(0, 6 * np.pi, 200)
        data = (40 + 30 * np.sin(t)).astype(np.float32)

        # Train
        model = WorkloadLSTM(input_size=1, hidden_size=16, horizon=3)
        model.fit_scaler(data)
        ds = SlidingWindowDataset(window_size=12, horizon=3)
        (X_tr, y_tr), _, _ = ds.split(model.normalize(data))
        model.train(X_tr, y_tr, epochs=10, lr=1e-3)

        # Predict on last window
        window = data[-12:]
        preds  = model.predict(window)
        assert preds.shape == (3,)

        # Scaling decision
        with tempfile.TemporaryDirectory() as tmp:
            controller = ASGController(log_path=tmp)
            policy = ScalingPolicy(
                asg_name="e2e-test-asg",
                scale_out_threshold=70.0,
                scale_in_threshold=30.0,
            )
            action = controller.decide(policy, float(preds[0]))
            assert action in ("SCALE_OUT", "SCALE_IN", "NO_ACTION")
