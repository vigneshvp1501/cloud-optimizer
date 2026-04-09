"""
CloudWatch Metric Publisher
Publishes 'PredictedLoad' custom metric to AWS CloudWatch
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MetricPublisher:
    """
    Publishes predicted workload metrics to CloudWatch under a custom namespace.
    These metrics drive the CloudWatch Alarms that trigger Auto Scaling.
    """

    NAMESPACE = "CloudOptimizer/PredictedLoad"

    def __init__(self, region: str = "us-east-1", namespace: Optional[str] = None):
        self.cw        = boto3.client("cloudwatch", region_name=region)
        self.namespace = namespace or self.NAMESPACE
        self.region    = region

    # ── Core publish ──────────────────────────
    def publish_predicted_cpu(
        self,
        value: float,
        asg_name: str,
        unit: str = "Percent",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Publishes a single PredictedLoad value.

        Args:
            value:    Predicted CPU utilisation (0-100)
            asg_name: Auto Scaling Group name (used as dimension)
            unit:     CloudWatch unit string
            timestamp: Optional explicit timestamp (defaults to now)
        """
        try:
            metric_data = {
                "MetricName": "PredictedCPUUtilization",
                "Dimensions": [
                    {"Name": "AutoScalingGroupName", "Value": asg_name}
                ],
                "Value":     max(0.0, min(100.0, value)),
                "Unit":      unit,
                "Timestamp": timestamp or datetime.utcnow(),
            }

            self.cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data],
            )

            logger.info(
                f"Published PredictedCPUUtilization={value:.2f}% "
                f"for ASG '{asg_name}' → {self.namespace}"
            )
            return True

        except ClientError as e:
            logger.error(f"CloudWatch publish failed: {e}")
            return False

    def publish_horizon(
        self,
        predictions: List[float],
        asg_name: str,
        interval_minutes: int = 60,
    ) -> int:
        """
        Publishes a full prediction horizon as timestamped datapoints.

        Args:
            predictions:       List of predicted values (one per horizon step)
            asg_name:          ASG name for the dimension
            interval_minutes:  Minutes between each horizon step
        Returns:
            Number of datapoints successfully published
        """
        success = 0
        now = datetime.utcnow()

        metric_data = []
        for i, value in enumerate(predictions):
            ts = datetime(
                now.year, now.month, now.day, now.hour
            )
            from datetime import timedelta
            ts += timedelta(minutes=interval_minutes * (i + 1))

            metric_data.append({
                "MetricName": "PredictedCPUUtilization",
                "Dimensions": [
                    {"Name": "AutoScalingGroupName", "Value": asg_name}
                ],
                "Value":     max(0.0, min(100.0, float(value))),
                "Unit":      "Percent",
                "Timestamp": ts,
            })

        # CloudWatch accepts max 20 datapoints per call
        for batch_start in range(0, len(metric_data), 20):
            batch = metric_data[batch_start : batch_start + 20]
            try:
                self.cw.put_metric_data(Namespace=self.namespace, MetricData=batch)
                success += len(batch)
            except ClientError as e:
                logger.error(f"Batch publish error: {e}")

        logger.info(f"Published {success}/{len(predictions)} horizon datapoints for '{asg_name}'")
        return success

    # ── Alarm management ──────────────────────
    def create_predictive_alarm(
        self,
        asg_name: str,
        scale_out_threshold: float = 70.0,
        scale_in_threshold: float  = 30.0,
        evaluation_periods: int    = 2,
        scale_out_action_arn: Optional[str] = None,
        scale_in_action_arn:  Optional[str] = None,
    ) -> dict:
        """
        Creates two CloudWatch Alarms (scale-out & scale-in) on the
        PredictedCPUUtilization custom metric.
        """
        results = {}

        for direction, threshold, action_arn, comparison in [
            ("ScaleOut", scale_out_threshold, scale_out_action_arn, "GreaterThanOrEqualToThreshold"),
            ("ScaleIn",  scale_in_threshold,  scale_in_action_arn,  "LessThanOrEqualToThreshold"),
        ]:
            alarm_name = f"CloudOptimizer-{asg_name}-{direction}"
            kwargs = dict(
                AlarmName=alarm_name,
                AlarmDescription=f"Predictive {direction} alarm for {asg_name}",
                Namespace=self.namespace,
                MetricName="PredictedCPUUtilization",
                Dimensions=[{"Name": "AutoScalingGroupName", "Value": asg_name}],
                Period=3600,
                EvaluationPeriods=evaluation_periods,
                Threshold=threshold,
                ComparisonOperator=comparison,
                Statistic="Average",
                TreatMissingData="notBreaching",
            )
            if action_arn:
                kwargs["AlarmActions"] = [action_arn]

            try:
                self.cw.put_metric_alarm(**kwargs)
                logger.info(f"Alarm '{alarm_name}' created/updated (threshold={threshold}%)")
                results[direction] = alarm_name
            except ClientError as e:
                logger.error(f"Failed to create alarm {alarm_name}: {e}")
                results[direction] = None

        return results

    def get_alarm_state(self, alarm_name: str) -> Optional[str]:
        try:
            resp = self.cw.describe_alarms(AlarmNames=[alarm_name])
            alarms = resp.get("MetricAlarms", [])
            if alarms:
                return alarms[0]["StateValue"]   # OK | ALARM | INSUFFICIENT_DATA
        except ClientError as e:
            logger.error(f"describe_alarms error: {e}")
        return None
