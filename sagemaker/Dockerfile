# Unified GPU inference container for SageMaker
# Contains all vision models: YOLO, SAM3, DINOv3, T-LEAP
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget git python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    ultralytics>=8.0.0 \
    segment-anything \
    transformers \
    flask \
    gunicorn \
    boto3 \
    pyyaml \
    numpy>=1.24.0 \
    opencv-python>=4.8.0 \
    Pillow

COPY sagemaker/ /app/sagemaker/
COPY shared/ /app/shared/

ENV PYTHONPATH=/app:$PYTHONPATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "1", "--timeout", "600", "sagemaker.serve:app"]
