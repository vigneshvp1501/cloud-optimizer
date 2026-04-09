# Cloud Resource Optimization System

Predictive auto-scaling using a Dockerized LSTM model, AWS CloudWatch custom metrics, and Auto Scaling Groups.

---

## Architecture

```
Cloud Infrastructure
       │ (metrics)
       ▼
 AWS CloudWatch ──────────────────────────────────────────────────────────┐
       │ Historical Logs                                                    │
       ▼                                                                    │
┌──────────────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  Monitoring &        │    │ Prediction Engine │    │  AWS CloudWatch   │ │
│  Data Storage        │───▶│                  │───▶│  (custom metric)  │ │
│  (CloudWatch + DB)   │    │ Dockerized LSTM   │    │  PredictedLoad    │ │
└──────────────────────┘    └──────────────────┘    └────────┬──────────┘ │
         │                          │                         │            │
         │ Triggers                 │ Sliding Window          │ Alarm      │
         ▼                          ▼                         ▼            │
┌──────────────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│ Partitioned Data /   │    │ CloudWatch Metric │    │  Decision &       │ │
│ Dockerized LSTM DB   │    │ Publishing        │    │  Alarm            │ │
└──────────────────────┘    └──────────────────┘    └────────┬──────────┘ │
                                                              │            │
                                                    ┌─────────▼──────────┐│
                                                    │  Auto Scaling Group ││
                                                    │  ● Scaling Actions  ││
                                                    │  ● Instance Count   ││
                                                    │  ● Predicted Values ││
                                                    │  ● Scaling Cost     │◀┘
                                                    └────────────────────┘
```

---

## Project Structure

```
cloud-optimizer/
├── model/
│   └── lstm_model.py          # Two-layer LSTM, sliding window, scaler
├── publisher/
│   └── metric_publisher.py    # CloudWatch PredictedLoad metric & alarms
├── scaler/
│   └── asg_controller.py      # ASG decision engine, cooldown, cost log
├── tests/
│   └── test_optimizer.py      # pytest suite (no AWS required for unit tests)
├── infra/
│   └── cloudformation.yaml    # Full AWS stack (ASG, ECS, alarms, IAM)
├── config/
│   └── config.json            # Runtime configuration
├── orchestrator.py            # Main loop: fetch → predict → publish → scale
├── Dockerfile                 # Production container
├── docker-compose.yml         # Local dev (+ LocalStack)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS credentials

```bash
aws configure
# or set environment variables:
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### 3. Edit config

```bash
vim config/config.json
# Set asg_name, region, thresholds, etc.
```

### 4. Run locally (dry run)

```bash
export DRY_RUN=true
python orchestrator.py
```

### 5. Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Docker Deployment

### Build & push to ECR

```bash
# Build
docker build -t cloud-optimizer .

# Tag
docker tag cloud-optimizer:latest \
  <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/cloud-optimizer:latest

# Push
aws ecr get-login-password | docker login --username AWS \
  --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/cloud-optimizer:latest
```

### Local dev with LocalStack

```bash
docker compose --profile dev up
```

---

## AWS Deployment (CloudFormation)

```bash
aws cloudformation deploy \
  --template-file infra/cloudformation.yaml \
  --stack-name cloud-optimizer \
  --parameter-overrides \
      VpcId=vpc-xxxxxxxx \
      SubnetIds=subnet-aaa,subnet-bbb \
      PredictorImageUri=<ECR_URI> \
      ASGName=my-autoscaling-group \
  --capabilities CAPABILITY_NAMED_IAM
```

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `region` | `us-east-1` | AWS region |
| `asg_name` | `my-autoscaling-group` | Target Auto Scaling Group |
| `metric_namespace` | `AWS/EC2` | CloudWatch namespace to read |
| `metric_name` | `CPUUtilization` | Metric to predict |
| `window_size` | `24` | Hours of history fed to LSTM |
| `horizon` | `6` | Hours ahead to predict |
| `poll_interval_sec` | `3600` | Loop interval (seconds) |
| `scale_out_threshold` | `70.0` | % CPU to trigger scale-out |
| `scale_in_threshold` | `30.0` | % CPU to trigger scale-in |
| `min_capacity` | `1` | ASG minimum instance count |
| `max_capacity` | `20` | ASG maximum instance count |
| `cooldown_seconds` | `300` | Scaling cooldown window |
| `instance_hourly_cost` | `0.096` | $/hr per instance (for cost logging) |
| `dry_run` | `false` | Log decisions without calling AWS |

All keys can be overridden via UPPER_CASE environment variables.

---

## How It Works

1. **Monitoring & Data Storage** – CloudWatch emits standard EC2/ASG metrics (CPUUtilization).
2. **Fetch** – The orchestrator pulls the last `window_size` hours of metrics via `GetMetricStatistics`.
3. **Predict** – The LSTM model processes the sliding window and outputs a `horizon`-step forecast.
4. **Publish** – Predictions are pushed to the `CloudOptimizer/PredictedLoad` custom CloudWatch namespace as `PredictedCPUUtilization`.
5. **Alarm** – Two CloudWatch Alarms (scale-out ≥ threshold, scale-in ≤ threshold) evaluate the custom metric.
6. **Scale** – The ASGController calls `SetDesiredCapacity` respecting min/max/cooldown policies.
7. **Log** – Every scaling event (action, capacities, cost delta) is written to a JSONL log.

---

## IAM Permissions Required

```json
{
  "Effect": "Allow",
  "Action": [
    "cloudwatch:GetMetricStatistics",
    "cloudwatch:PutMetricData",
    "cloudwatch:PutMetricAlarm",
    "cloudwatch:DescribeAlarms",
    "autoscaling:DescribeAutoScalingGroups",
    "autoscaling:SetDesiredCapacity"
  ],
  "Resource": "*"
}
```
