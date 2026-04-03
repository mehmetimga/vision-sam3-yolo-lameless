# SageMaker GPU Inference Setup

## Overview

This project supports two GPU deployment modes:

| Mode | Cost Model | Best For | Toggle |
|------|-----------|----------|--------|
| **EC2 GPU Worker** | Always-on (Spot or On-Demand) | Heavy/continuous usage | `gpu_enabled = true` |
| **SageMaker Async** | Pay-per-use (scale to zero) | Light/sporadic usage | `sagemaker_enabled = true` |

## Architecture

### EC2 Mode (Default)
```
NATS → GPU EC2 (8 containers, shared GPU) → Results to EFS → NATS
```

### SageMaker Mode
```
NATS → sagemaker-bridge (Fargate CPU)
         ├── Upload video to S3
         ├── Call SageMaker endpoint (GPU scales up)
         ├── Poll for results
         ├── Save results to EFS
         └── Publish to NATS
```

All 4 vision models (YOLO, SAM3, DINOv3, T-LEAP) run in a single SageMaker
endpoint behind one GPU instance. When idle, the endpoint scales to zero.

## Cost Comparison

| Scenario | EC2 Spot 24/7 | SageMaker (2h/day) | SageMaker (6h/day) |
|----------|--------------|--------------------|--------------------|
| GPU cost | ~$117/mo | ~$44/mo | ~$133/mo |
| Base infra | ~$246/mo | ~$260/mo | ~$260/mo |
| **Total** | **~$363/mo** | **~$304/mo** | **~$393/mo** |

SageMaker breaks even at ~4-5 hours of GPU usage per day.

## Deployment Steps

### 1. Build SageMaker Images

```bash
# Build and push the unified GPU inference image + bridge service
./scripts/build-sagemaker-images.sh

# Or with custom tag
TAG=v1.0 ./scripts/build-sagemaker-images.sh
```

This builds two images:
- `gpu-inference:latest` — All vision models (YOLO, SAM3, DINOv3, T-LEAP)
- `sagemaker-bridge:latest` — Lightweight NATS-to-SageMaker orchestrator

### 2. Enable SageMaker in Terraform

Edit `terraform/terraform.tfvars`:

```hcl
# Disable EC2 GPU worker (optional, SageMaker overrides it automatically)
gpu_enabled = false

# Enable SageMaker
sagemaker_enabled       = true
sagemaker_instance_type = "ml.g4dn.xlarge"
sagemaker_max_instances = 1
```

### 3. Apply Terraform

```bash
cd terraform
terraform plan    # Review changes
terraform apply   # Apply
```

This will:
- Create SageMaker model, endpoint config, and endpoint
- Create S3 bucket for async I/O
- Create SNS topic for notifications
- Set up auto-scaling (0 to 1 instances)
- Add `sagemaker-bridge` Fargate service to ECS
- Disable EC2 GPU ASG (desired = 0)

### 4. Verify

```bash
# Check SageMaker endpoint status
aws sagemaker describe-endpoint \
    --endpoint-name cow-lameness-production-gpu-inference \
    --query 'EndpointStatus'

# Check bridge service is running
aws ecs describe-services \
    --cluster cow-lameness-production \
    --services sagemaker-bridge \
    --query 'services[0].runningCount'
```

## How It Works

1. **Video uploaded** → preprocessing → NATS `video.preprocessed` event
2. **sagemaker-bridge** (Fargate) picks up the event
3. Bridge uploads processed video from EFS to S3
4. Bridge calls SageMaker `InvokeEndpointAsync` with `pipeline: "all"`
5. SageMaker endpoint **scales from 0 → 1** GPU instance (~5 min cold start)
6. Unified container runs YOLO → SAM3 → DINOv3 → T-LEAP sequentially
7. Results returned via S3
8. Bridge saves results to EFS and publishes to NATS
9. Downstream services (TCN, Transformer, etc.) process as normal
10. After idle period (~10 min), endpoint **scales back to 0**

## Cold Start Behavior

When the endpoint is at 0 instances:
- First request takes **5-10 minutes** (instance launch + model loading)
- Subsequent requests (within cooldown window) are processed immediately
- After 10 minutes of idle, scales back to 0

For light usage (1-5 videos/day), expect ~5 min latency on the first video
of the day. Subsequent videos within the session process in 1-2 minutes.

## Switching Between Modes

### Switch to SageMaker
```hcl
sagemaker_enabled = true   # GPU worker is automatically disabled
```

### Switch back to EC2
```hcl
sagemaker_enabled = false
gpu_enabled       = true
use_spot_instances = true
```

Both modes can coexist in the same Terraform state. Only one is active at a time.

## File Structure

```
sagemaker/
├── Dockerfile           # Unified GPU container (all vision models)
├── serve.py             # Flask inference server (routes to handlers)
└── handlers/
    ├── yolo.py          # YOLO detection handler
    ├── sam3.py          # SAM3 segmentation handler
    ├── dinov3.py        # DINOv3 embedding handler
    └── tleap.py         # T-LEAP pose estimation handler

services/sagemaker-bridge/
├── Dockerfile           # Lightweight Fargate bridge
└── app/main.py          # NATS → SageMaker orchestrator

shared/utils/
└── sagemaker_client.py  # Async inference client with polling

terraform/modules/sagemaker/
├── main.tf              # SageMaker resources
├── variables.tf
└── outputs.tf
```
