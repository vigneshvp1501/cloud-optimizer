"""
Auto Scaling Group Controller
Handles scaling decisions, executes scale-out/in actions, and logs outcomes.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional, Dict
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class ScalingEvent:
    timestamp:      str
    asg_name:       str
    action:         str           # SCALE_OUT | SCALE_IN | NO_ACTION
    reason:         str
    old_capacity:   int
    new_capacity:   int
    predicted_load: float
    estimated_cost: float = 0.0
    success:        bool  = True


@dataclass
class ScalingPolicy:
    asg_name:              str
    min_capacity:          int   = 1
    max_capacity:          int   = 20
    desired_capacity:      int   = 2
    scale_out_threshold:   float = 70.0   # % CPU
    scale_in_threshold:    float = 30.0
    scale_out_step:        int   = 2      # instances to add
    scale_in_step:         int   = 1      # instances to remove
    cooldown_seconds:      int   = 300
    instance_hourly_cost:  float = 0.096  # $/hr  (e.g. m5.large)


# ─────────────────────────────────────────────
# ASG Controller
# ─────────────────────────────────────────────
class ASGController:
    """
    Wraps AWS Auto Scaling API.
    Decides and executes scaling actions based on predicted CPU load,
    respects min/max boundaries and cooldown windows.
    """

    def __init__(self, region: str = "us-east-1", log_path: str = "./logs"):
        self.asg    = boto3.client("autoscaling",    region_name=region)
        self.ec2    = boto3.client("ec2",            region_name=region)
        self.region = region
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self._last_scale: Dict[str, datetime] = {}

    # ── Current state ─────────────────────────
    def get_current_capacity(self, asg_name: str) -> Optional[int]:
        try:
            resp = self.asg.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
            groups = resp.get("AutoScalingGroups", [])
            if groups:
                return groups[0]["DesiredCapacity"]
        except ClientError as e:
            logger.error(f"describe_auto_scaling_groups error: {e}")
        return None

    def get_running_instances(self, asg_name: str) -> List[str]:
        try:
            resp = self.asg.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
            groups = resp.get("AutoScalingGroups", [])
            if groups:
                return [i["InstanceId"] for i in groups[0].get("Instances", [])
                        if i["LifecycleState"] == "InService"]
        except ClientError:
            pass
        return []

    # ── Decision engine ───────────────────────
    def decide(self, policy: ScalingPolicy, predicted_cpu: float) -> str:
        """Return SCALE_OUT | SCALE_IN | NO_ACTION."""
        if predicted_cpu >= policy.scale_out_threshold:
            return "SCALE_OUT"
        if predicted_cpu <= policy.scale_in_threshold:
            return "SCALE_IN"
        return "NO_ACTION"

    def _in_cooldown(self, asg_name: str, cooldown: int) -> bool:
        last = self._last_scale.get(asg_name)
        if last is None:
            return False
        elapsed = (datetime.utcnow() - last).total_seconds()
        return elapsed < cooldown

    # ── Execute scaling ────────────────────────
    def scale(
        self,
        policy: ScalingPolicy,
        predicted_cpu: float,
        dry_run: bool = False,
    ) -> ScalingEvent:
        """
        Evaluates predicted load against policy thresholds and scales accordingly.

        Args:
            policy:        Scaling policy definition
            predicted_cpu: Predicted CPU utilisation (%)
            dry_run:       If True, log only – do not call AWS API
        """
        action = self.decide(policy, predicted_cpu)
        current = self.get_current_capacity(policy.asg_name) or policy.desired_capacity

        new_capacity = current
        if action == "SCALE_OUT":
            new_capacity = min(current + policy.scale_out_step, policy.max_capacity)
        elif action == "SCALE_IN":
            new_capacity = max(current - policy.scale_in_step, policy.min_capacity)

        reason = (
            f"PredictedCPU={predicted_cpu:.1f}% "
            f"(thresholds: out≥{policy.scale_out_threshold}%, in≤{policy.scale_in_threshold}%)"
        )

        cost_delta = (new_capacity - current) * policy.instance_hourly_cost
        event = ScalingEvent(
            timestamp=datetime.utcnow().isoformat(),
            asg_name=policy.asg_name,
            action=action,
            reason=reason,
            old_capacity=current,
            new_capacity=new_capacity,
            predicted_load=predicted_cpu,
            estimated_cost=cost_delta,
        )

        if action == "NO_ACTION" or current == new_capacity:
            logger.info(f"[{policy.asg_name}] NO_ACTION — {reason}")
            self._log_event(event)
            return event

        if self._in_cooldown(policy.asg_name, policy.cooldown_seconds):
            logger.warning(f"[{policy.asg_name}] Cooldown active – skipping {action}")
            event.action = "NO_ACTION (cooldown)"
            event.new_capacity = current
            self._log_event(event)
            return event

        if not dry_run:
            try:
                self.asg.set_desired_capacity(
                    AutoScalingGroupName=policy.asg_name,
                    DesiredCapacity=new_capacity,
                    HonorCooldown=False,       # we manage cooldown ourselves
                )
                self._last_scale[policy.asg_name] = datetime.utcnow()
                logger.info(
                    f"[{policy.asg_name}] {action}: {current} → {new_capacity} "
                    f"(ΔCost ${cost_delta:+.3f}/hr)"
                )
            except ClientError as e:
                logger.error(f"set_desired_capacity failed: {e}")
                event.success = False
        else:
            logger.info(f"[DRY RUN] Would {action}: {current} → {new_capacity}")

        self._log_event(event)
        return event

    # ── Logging ───────────────────────────────
    def _log_event(self, event: ScalingEvent):
        log_file = os.path.join(
            self.log_path,
            f"scaling_{event.asg_name}_{datetime.utcnow():%Y%m}.jsonl"
        )
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def load_history(self, asg_name: str) -> List[ScalingEvent]:
        """Read all logged scaling events for an ASG."""
        events = []
        for fname in sorted(os.listdir(self.log_path)):
            if fname.startswith(f"scaling_{asg_name}_"):
                with open(os.path.join(self.log_path, fname)) as f:
                    for line in f:
                        try:
                            events.append(ScalingEvent(**json.loads(line)))
                        except Exception:
                            pass
        return events

    # ── Cost reporting ────────────────────────
    def cost_report(self, asg_name: str) -> dict:
        history = self.load_history(asg_name)
        total_instances_hours = sum(
            e.new_capacity for e in history if e.success
        )
        scale_outs = sum(1 for e in history if e.action == "SCALE_OUT")
        scale_ins  = sum(1 for e in history if e.action == "SCALE_IN")
        return {
            "asg_name":          asg_name,
            "total_events":      len(history),
            "scale_outs":        scale_outs,
            "scale_ins":         scale_ins,
            "estimated_cost_usd": round(
                sum(e.estimated_cost for e in history if e.success), 4
            ),
        }
