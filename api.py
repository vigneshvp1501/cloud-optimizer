import sys
import os

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lstm_model import WorkloadLSTM, predict_next
import numpy as np
import random
from collections import deque
from datetime import datetime

# ✅ NEW IMPORTS (REAL SCALING)
from asg_controller import ASGController, ScalingPolicy

# ─────────────────────────────────────────────
# GLOBAL STATE (ONLY FOR PREDICTION, NOT SCALING)
# ─────────────────────────────────────────────
history = deque([60, 62, 65, 63, 67, 70, 68, 72, 75, 78], maxlen=20)
prev_prediction = 70

# ─────────────────────────────────────────────
# MODEL INITIALIZATION
# ─────────────────────────────────────────────
model = WorkloadLSTM(input_size=1, hidden_size=64, horizon=6)
model.fit_scaler(np.array([10, 20, 30, 40, 50], dtype=np.float32))

# ─────────────────────────────────────────────
# AWS ASG CONTROLLER (REAL SCALING)
# ─────────────────────────────────────────────
controller = ASGController(region="us-east-1")

policy = ScalingPolicy(
    asg_name="my-autoscaling-group",   # ⚠️ CHANGE THIS if needed
    min_capacity=2,
    max_capacity=12,
    desired_capacity=2,
    scale_out_threshold=70.0,
    scale_in_threshold=30.0
)

# ─────────────────────────────────────────────
# FASTAPI SETUP
# ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Backend is running (REAL SCALING ENABLED)"}


# ─────────────────────────────────────────────
# 🔥 MANUAL SCALING (REAL AWS)
# ─────────────────────────────────────────────
@app.post("/manual-scale")
def manual_scale(instance_count: int):
    controller.asg.set_desired_capacity(
        AutoScalingGroupName=policy.asg_name,
        DesiredCapacity=instance_count,
        HonorCooldown=False
    )
    return {
        "message": f"Scaled to {instance_count}",
        "mode": "manual"
    }


# ─────────────────────────────────────────────
# 🔥 AUTO SCALING STATUS (REAL AWS)
# ─────────────────────────────────────────────
@app.get("/asg-status")
def asg_status():
    global prev_prediction
    global history

    # ─────────────────────────────
    # Simulate workload (for demo)
    # ─────────────────────────────
    last = history[-1]
    new_value = max(30, min(95, last + random.uniform(-3, 5)))
    history.append(new_value)

    recent_values = list(history)[-10:]

    # ─────────────────────────────
    # Prediction
    # ─────────────────────────────
    preds = predict_next(model, recent_values)
    raw_pred = float(preds[0])

    # smoothing
    predicted = 0.7 * prev_prediction + 0.3 * raw_pred
    prev_prediction = predicted

    # ─────────────────────────────
    # REAL AUTO SCALING (AWS)
    # ─────────────────────────────
    event = controller.scale(
        policy=policy,
        predicted_cpu=predicted
    )

    current_instances = event.new_capacity

    print("Pred:", round(predicted, 2), "| Instances:", current_instances)

    return {
        "running": current_instances,
        "min": policy.min_capacity,
        "max": policy.max_capacity,
        "predicted": round(predicted, 2),
        "action": event.action,
        "timestamp": datetime.now().isoformat()
    }