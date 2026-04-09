# ── Build stage ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────
FROM python:3.11-slim

LABEL maintainer="cloud-optimizer"
LABEL description="Dockerized LSTM Workload Prediction & Auto Scaling Engine"

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY model/       ./model/
COPY publisher/   ./publisher/
COPY scaler/      ./scaler/
COPY orchestrator.py .

# Create runtime directories
RUN mkdir -p /app/logs /app/config /app/data

# Non-root user for security
RUN useradd -r -u 1001 optimizer && chown -R optimizer:optimizer /app
USER optimizer

# Health check – verify Python can import core modules
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
  CMD python -c "from model.lstm_model import WorkloadLSTM; print('healthy')" || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "orchestrator.py"]
