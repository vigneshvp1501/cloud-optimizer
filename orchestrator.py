"""
Cloud Resource Optimization Orchestrator
Main loop: fetch → predict → publish → scale
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Optional
import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from model.lstm_model import WorkloadLSTM, SlidingWindowDataset, CloudWatchFetcher
from publisher.metric_publisher import MetricPublisher
from scaler.asg_controller import ASGController, ScalingPolicy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/orchestrator.log"),
    ],
)


# ─────────────────────────────────────────────
# Config (reads from environment / file)
# ─────────────────────────────────────────────
class Config:
    def __init__(self, path: str = "/app/config/config.json"):
        defaults = {
            "region":               "us-east-1",
            "metric_namespace":     "AWS/EC2",
            "metric_name":          "CPUUtilization",
            "asg_name":             "my-autoscaling-group",
            "window_size":          24,
            "horizon":              6,
            "poll_interval_sec":    3600,
            "model_path":           "/app/model/lstm_weights.json",
            "scale_out_threshold":  70.0,
            "scale_in_threshold":   30.0,
            "min_capacity":         1,
            "max_capacity":         20,
            "cooldown_seconds":     300,
            "instance_hourly_cost": 0.096,
            "dry_run":              False,
        }

        if os.path.exists(path):
            with open(path) as f:
                overrides = json.load(f)
            defaults.update(overrides)

        # Environment variable overrides (UPPER_CASE)
        for key in defaults:
            env_key = key.upper()
            if env_val := os.environ.get(env_key):
                if isinstance(defaults[key], bool):
                    defaults[key] = env_val.lower() in ("1", "true", "yes")
                elif isinstance(defaults[key], int):
                    defaults[key] = int(env_val)
                elif isinstance(defaults[key], float):
                    defaults[key] = float(env_val)
                else:
                    defaults[key] = env_val

        self.__dict__.update(defaults)
        logger.info(f"Config loaded: ASG={self.asg_name}  Region={self.region}")


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
class Orchestrator:
    """
    Full pipeline:
      1. Fetch recent CloudWatch metrics
      2. Run LSTM prediction
      3. Publish PredictedLoad custom metric
      4. Evaluate alarms
      5. Execute Auto Scaling decision
      6. Log outcome
    """

    def __init__(self, config: Config):
        self.cfg       = config
        self.fetcher   = CloudWatchFetcher(region=config.region)
        self.publisher = MetricPublisher(region=config.region)
        self.scaler    = ASGController(region=config.region)
        self.model: Optional[WorkloadLSTM] = None
        self.policy    = ScalingPolicy(
            asg_name=config.asg_name,
            min_capacity=config.min_capacity,
            max_capacity=config.max_capacity,
            scale_out_threshold=config.scale_out_threshold,
            scale_in_threshold=config.scale_in_threshold,
            cooldown_seconds=config.cooldown_seconds,
            instance_hourly_cost=config.instance_hourly_cost,
        )

    # ── Model bootstrap ───────────────────────
    def ensure_model(self):
        """Load existing weights or train from scratch."""
        if os.path.exists(self.cfg.model_path):
            self.model = WorkloadLSTM.load(self.cfg.model_path)
        else:
            logger.info("No saved model found — training from CloudWatch history …")
            self.model = self._train_new_model()

    def _train_new_model(self) -> WorkloadLSTM:
        _, values = self.fetcher.fetch(
            namespace=self.cfg.metric_namespace,
            metric_name=self.cfg.metric_name,
            dimensions=self._dimensions(),
            hours=720,
        )

        if len(values) < self.cfg.window_size + self.cfg.horizon + 10:
            logger.warning("Insufficient history — using synthetic data for bootstrap")
            values = self._synthetic_data(length=500)

        data  = np.array(values, dtype=np.float32)
        model = WorkloadLSTM(
            input_size=1,
            hidden_size=64,
            horizon=self.cfg.horizon,
        )
        model.fit_scaler(data)

        dataset = SlidingWindowDataset(self.cfg.window_size, self.cfg.horizon)
        (X_tr, y_tr), _, _ = dataset.split(model.normalize(data))
        model.train(X_tr, y_tr, epochs=80, lr=5e-4)
        model.save(self.cfg.model_path)
        return model

    # ── Main tick ─────────────────────────────
    def tick(self) -> dict:
        """Execute one prediction-scaling cycle. Returns a summary dict."""
        logger.info("=" * 60)
        logger.info(f"Tick at {datetime.utcnow().isoformat()}Z")

        # 1. Fetch recent window of metrics
        _, recent = self.fetcher.fetch(
            namespace=self.cfg.metric_namespace,
            metric_name=self.cfg.metric_name,
            dimensions=self._dimensions(),
            hours=self.cfg.window_size,
        )

        if len(recent) < self.cfg.window_size:
            logger.warning(
                f"Only {len(recent)}/{self.cfg.window_size} datapoints available — "
                "padding with zeros"
            )
            recent = [0.0] * (self.cfg.window_size - len(recent)) + list(recent)

        # 2. Predict
        predictions = self.model.predict(np.array(recent, dtype=np.float32))
        next_hour_prediction = float(predictions[0])
        logger.info(f"Predictions (next {self.cfg.horizon}h): "
                    f"{[f'{p:.1f}%' for p in predictions]}")

        # 3. Publish to CloudWatch
        self.publisher.publish_predicted_cpu(
            value=next_hour_prediction,
            asg_name=self.cfg.asg_name,
        )
        self.publisher.publish_horizon(
            predictions=predictions.tolist(),
            asg_name=self.cfg.asg_name,
        )

        # 4. Make scaling decision
        event = self.scaler.scale(
            policy=self.policy,
            predicted_cpu=next_hour_prediction,
            dry_run=self.cfg.dry_run,
        )

        summary = {
            "timestamp":       datetime.utcnow().isoformat(),
            "asg":             self.cfg.asg_name,
            "predictions":     [round(float(p), 2) for p in predictions],
            "next_hour_cpu":   round(next_hour_prediction, 2),
            "action":          event.action,
            "old_capacity":    event.old_capacity,
            "new_capacity":    event.new_capacity,
            "estimated_cost":  event.estimated_cost,
        }

        logger.info(f"Tick complete: {json.dumps(summary, indent=2)}")
        return summary

    # ── Run loop ──────────────────────────────
    def run(self):
        """Continuous prediction-scaling loop."""
        logger.info("Cloud Resource Optimizer starting …")
        self.ensure_model()

        # Create alarms on first run
        self.publisher.create_predictive_alarm(
            asg_name=self.cfg.asg_name,
            scale_out_threshold=self.cfg.scale_out_threshold,
            scale_in_threshold=self.cfg.scale_in_threshold,
        )

        while True:
            try:
                self.tick()
            except Exception as e:
                logger.exception(f"Tick failed: {e}")

            logger.info(f"Sleeping {self.cfg.poll_interval_sec}s …")
            time.sleep(self.cfg.poll_interval_sec)

    # ── Helpers ───────────────────────────────
    def _dimensions(self) -> List[dict]:
        return [{"Name": "AutoScalingGroupName", "Value": self.cfg.asg_name}]

    @staticmethod
    def _synthetic_data(length: int = 500) -> List[float]:
        """Generate sine-wave workload for bootstrapping without real data."""
        t = np.linspace(0, 10 * np.pi, length)
        return (40 + 30 * np.sin(t) + np.random.normal(0, 5, length)).tolist()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    orch = Orchestrator(cfg)
    orch.run()
