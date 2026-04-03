# SageMaker Bridge - lightweight Fargate service (no GPU needed)
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    nats-py>=2.6.0 \
    pyyaml>=6.0.1 \
    boto3 \
    numpy>=1.24.0

COPY services/sagemaker-bridge/ /app/
COPY shared/ /app/shared/

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "app/main.py"]
