# AWS Infrastructure & Terraform Guide

## Overview

The entire AWS infrastructure is managed via Terraform in the `terraform/` directory. All resources are created in **us-west-2** under the project name **cow-lameness-production**.

**Production URL:** https://cowhealth.ai

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     Internet                            │
                    └──────────────────────┬──────────────────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │   Route53   │  cowhealth.ai
                                    │   DNS       │  www.cowhealth.ai
                                    └──────┬──────┘  api.cowhealth.ai
                                           │
                              ┌────────────▼────────────┐
                              │  ALB (HTTPS, port 443)  │  ACM Certificate
                              │  Public Subnets         │
                              ├─────────┬───────────────┤
                              │ /* → FE │ /api/* → BE   │
                              └────┬────┴──────┬────────┘
                                   │           │
               ┌───────────────────▼───────────▼───────────────────┐
               │              Private Subnets (10.0.10-11.x)       │
               │                                                    │
               │  ┌─────────────────────────────────────────────┐  │
               │  │           ECS Fargate Cluster                │  │
               │  │                                              │  │
               │  │  admin-frontend ─── admin-backend            │  │
               │  │  nats ─── qdrant                             │  │
               │  │  video-ingestion ─── video-preprocessing     │  │
               │  │  clip-curation ─── tracking-service          │  │
               │  │  ml-pipeline ─── fusion-service              │  │
               │  │  annotation-renderer                         │  │
               │  │  sagemaker-bridge  ◄── (when SageMaker mode) │  │
               │  └──────────────────────────────────────────────┘  │
               │                                                    │
               │  ┌──────────────┐  ┌──────────────────────────┐   │
               │  │ NAT Gateway  │  │  VPC Endpoints           │   │
               │  │ (outbound)   │  │  S3, ECR API/DKR, Logs   │   │
               │  └──────────────┘  └──────────────────────────┘   │
               │                                                    │
               │  ┌──────────────────────────────────────────────┐ │
               │  │  GPU Compute (one of two modes):              │ │
               │  │                                               │ │
               │  │  Mode A: EC2 GPU Worker (g4dn.xlarge Spot)   │ │
               │  │    └── ASG 0-1, 8 containers, shared GPU     │ │
               │  │                                               │ │
               │  │  Mode B: SageMaker Async Inference            │ │
               │  │    └── ml.g4dn.xlarge, scale-to-zero          │ │
               │  │    └── S3 bucket for async I/O                │ │
               │  └──────────────────────────────────────────────┘ │
               │                                                    │
               │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
               │  │   EFS    │  │ S3 Vids  │  │ S3 Backups   │    │
               │  │ /data    │  │          │  │              │    │
               │  └──────────┘  └──────────┘  └──────────────┘    │
               └───────────────────────────────────────────────────┘
                                       │
               ┌───────────────────────▼───────────────────────────┐
               │           Database Subnets (10.0.20-21.x)         │
               │                                                    │
               │  ┌──────────────────────────────────────────────┐ │
               │  │  RDS PostgreSQL 15  (db.t4g.micro)           │ │
               │  │  cow_lameness database, 20 GB gp3            │ │
               │  └──────────────────────────────────────────────┘ │
               └───────────────────────────────────────────────────┘
```

## Terraform Modules

```
terraform/
├── main.tf                  # Root module — wires all modules together
├── variables.tf             # Input variables
├── outputs.tf               # Output values (ALB DNS, endpoints, etc.)
├── terraform.tfvars         # Environment-specific values
└── modules/
    ├── networking/          # VPC, subnets, NAT, security groups, VPC endpoints
    ├── ecs/                 # ECS Fargate cluster, task definitions, services
    ├── database/            # RDS PostgreSQL instance
    ├── storage/             # EFS file system, S3 buckets, CloudFront (optional)
    ├── load_balancer/       # Application Load Balancer, target groups, HTTPS
    ├── secrets/             # AWS Secrets Manager (DB creds, JWT, app config)
    ├── gpu_worker/          # EC2 Auto Scaling Group with GPU (optional)
    └── sagemaker/           # SageMaker async inference endpoint (optional)
```

### Module Details

#### networking

Creates the VPC and all network infrastructure.

| Resource | Details |
|----------|---------|
| VPC | `10.0.0.0/16` |
| Public subnets | `10.0.1.0/24`, `10.0.2.0/24` (2 AZs) |
| Private subnets | `10.0.10.0/24`, `10.0.11.0/24` |
| Database subnets | `10.0.20.0/24`, `10.0.21.0/24` |
| NAT Gateway | 1x in first public subnet (outbound for private subnets) |
| Internet Gateway | Attached to VPC for public subnets |
| VPC Endpoints | S3 (gateway), ECR API, ECR DKR, CloudWatch Logs (interface) |

**Security Groups:**

| SG | Ingress | Used By |
|----|---------|---------|
| ALB | 80, 443 from `0.0.0.0/0` | Load Balancer |
| ECS | From ALB + self + GPU SG | Fargate tasks |
| GPU | SSH from VPC CIDR, self | EC2 GPU instances, SageMaker |
| DB | 5432 from ECS + GPU | RDS |
| EFS | 2049 from ECS + GPU | EFS mount targets |

#### ecs

Manages the ECS Fargate cluster and all containerized services.

| Config | Value |
|--------|-------|
| Cluster | `cow-lameness-production-cluster` |
| Capacity | `FARGATE_SPOT` (default), `FARGATE` (base 1) |
| Service Discovery | `cow-lameness-production.local` (Cloud Map) |
| Logging | CloudWatch `/ecs/cow-lameness-production` |

**Services (12 when SageMaker enabled):**

| Service | CPU | Memory | Port | Notes |
|---------|-----|--------|------|-------|
| admin-backend | 2048 | 4096 | 8000 | FastAPI, registered with ALB |
| admin-frontend | 256 | 512 | 3000 | React, registered with ALB |
| nats | 256 | 512 | 4222 | Third-party image |
| qdrant | 1024 | 2048 | 6333 | Third-party image |
| video-ingestion | 512 | 1024 | 8001 | |
| video-preprocessing | 1024 | 2048 | 8002 | 50 GB ephemeral |
| clip-curation | 512 | 1024 | 8003 | |
| tracking-service | 512 | 1024 | 8004 | 50 GB ephemeral |
| ml-pipeline | 1024 | 2048 | 8005 | 50 GB ephemeral |
| fusion-service | 512 | 1024 | 8006 | 50 GB ephemeral |
| annotation-renderer | 1024 | 2048 | 8000 | |
| sagemaker-bridge | 512 | 1024 | 8007 | Only when `sagemaker_enabled` |

All services mount EFS at `/app/data` and get NATS/Qdrant URLs via environment variables. Secrets (DB URL, JWT) injected from Secrets Manager.

#### database

| Config | Value |
|--------|-------|
| Engine | PostgreSQL 15 |
| Instance | `db.t4g.micro` |
| Storage | 20 GB gp3 (auto-scales to 100 GB) |
| Database | `cow_lameness` |
| User | `postgres` |
| Backups | 7-day retention |
| Encryption | Enabled |
| Public | No (database subnets only) |

#### storage

**EFS (Elastic File System):**
- Encrypted, general purpose, bursting throughput
- Access points: `/data`, `/data/videos`, `/data/results`, `/data/models`
- Lifecycle: transition to IA after 30 days
- Mounted by all ECS tasks and GPU instances at `/app/data`

**S3 Buckets:**

| Bucket | Purpose | Lifecycle |
|--------|---------|-----------|
| `cow-lameness-production-videos-*` | Raw video uploads | IA after 90 days |
| `cow-lameness-production-backups-*` | Database/result backups | Glacier after 90 days |
| `cow-lameness-production-sagemaker-io` | SageMaker async I/O | Input: 3 days, output: 7 days |

**CloudFront** (optional, disabled by default): PriceClass_100 for video streaming.

#### load_balancer

| Config | Value |
|--------|-------|
| Type | Application (internet-facing) |
| Subnets | Public |
| HTTP (80) | Redirects to HTTPS |
| HTTPS (443) | TLS 1.3, ACM certificate for cowhealth.ai |
| Default | → admin-frontend |
| `/api/*` | → admin-backend |
| `/ws/*` | → admin-backend |

#### secrets

Three secrets in AWS Secrets Manager:
- `cow-lameness-production/database` — DB connection details
- `cow-lameness-production/jwt` — JWT signing key
- `cow-lameness-production/app-config` — DATABASE_URL, JWT_SECRET, NATS_URL, QDRANT_URL

ECS tasks reference `app-config` for injected environment variables.

#### gpu_worker (Mode A — EC2)

Always-on GPU instance for heavy/continuous usage.

| Config | Value |
|--------|-------|
| AMI | Deep Learning OSS Nvidia Driver AMI (PyTorch 2.3.1, AL2) |
| Instance | `g4dn.xlarge` (NVIDIA T4, 16 GB VRAM) |
| Spot | Enabled (70% savings) |
| ASG | 0-1 instances |
| EBS | 100 GB gp3 |
| Fallback pool | g4dn.xlarge, g4dn.2xlarge, g5.2xlarge |

Runs 8 GPU containers (YOLO, SAM3, DINOv3, T-LEAP, TCN, Transformer, GNN, Graph-Transformer) via Docker Compose on the instance. Subscribes to NATS for work.

**Disabled when `sagemaker_enabled = true`.**

#### sagemaker (Mode B — Pay-per-use)

On-demand GPU inference with scale-to-zero. Created when `sagemaker_enabled = true`.

| Config | Value |
|--------|-------|
| Endpoint | `cow-lameness-production-gpu-inference` |
| Instance | `ml.g4dn.xlarge` |
| Container | `gpu-inference:latest` (all 4 vision models) |
| Scaling | 0 → 1 instances (backlog-based) |
| Scale-in cooldown | 600 seconds (10 min idle → scale to 0) |
| Async I/O | S3 bucket `cow-lameness-production-sagemaker-io` |

The `sagemaker-bridge` Fargate service subscribes to NATS, uploads videos to S3, invokes SageMaker, and publishes results back to NATS.

## GPU Mode Comparison

| | EC2 GPU (Mode A) | SageMaker (Mode B) |
|--|-------------------|---------------------|
| **Toggle** | `gpu_enabled = true` | `sagemaker_enabled = true` |
| **Cost (light use)** | ~$360/mo (Spot 24/7) | ~$250-300/mo |
| **Cost (no use)** | ~$360/mo (still running) | ~$200/mo (GPU at $0) |
| **Latency** | Instant | 5-10 min cold start |
| **Scaling** | Manual (0 or 1) | Automatic (0 → 1 → 0) |
| **Management** | Docker Compose on EC2 | Fully managed by AWS |

## Key Terraform Variables

```hcl
# terraform/terraform.tfvars

project_name = "cow-lameness"
environment  = "production"
aws_region   = "us-west-2"

ecr_registry    = "703582588105.dkr.ecr.us-west-2.amazonaws.com"
certificate_arn = "arn:aws:acm:us-west-2:..."

# GPU Mode A: EC2 (set sagemaker_enabled = false)
gpu_enabled        = false
gpu_instance_type  = "g4dn.xlarge"
use_spot_instances = true

# GPU Mode B: SageMaker (set gpu_enabled = false)
sagemaker_enabled       = true
sagemaker_instance_type = "ml.g4dn.xlarge"
sagemaker_max_instances = 1
```

## Monthly Cost Breakdown

### Current (SageMaker mode, light usage)

| Component | Monthly |
|-----------|---------|
| Fargate (12 services, mostly Spot) | ~$130 |
| NAT Gateway | ~$35 |
| ALB | ~$20 |
| RDS (db.t4g.micro) | ~$15 |
| EFS | ~$20 |
| CloudWatch, Secrets, VPC Endpoints | ~$20 |
| SageMaker GPU (2-4 hrs/day) | ~$45-90 |
| **Total** | **~$285-330/mo** |

### Shutdown state (all services stopped)

| Component | Monthly |
|-----------|---------|
| NAT Gateway (if left on) | ~$35 |
| RDS (if not stopped) | ~$15 |
| EFS (minimal) | ~$5 |
| ALB (if left on) | ~$20 |
| **Total** | **~$25-75/mo** |

## Common Operations

### Deploy / Update

```bash
cd terraform
terraform plan     # Preview changes
terraform apply    # Apply changes
```

### Switch GPU Mode

```bash
# Edit terraform.tfvars, then:
terraform apply
```

### View Service Logs

```bash
# ECS service logs
aws logs tail /ecs/cow-lameness-production --filter-pattern "sagemaker-bridge" --follow

# SageMaker endpoint logs
aws logs tail /aws/sagemaker/Endpoints/cow-lameness-production-gpu-inference --follow
```

### Check Service Health

```bash
# All ECS services
aws ecs describe-services --cluster cow-lameness-production-cluster \
    --services admin-frontend admin-backend nats sagemaker-bridge \
    --query 'services[*].{name:serviceName,running:runningCount}'

# SageMaker endpoint
aws sagemaker describe-endpoint \
    --endpoint-name cow-lameness-production-gpu-inference \
    --query 'EndpointStatus'
```

### Scale Services

```bash
# Scale a service to 0 (stop)
aws ecs update-service --cluster cow-lameness-production-cluster \
    --service ml-pipeline --desired-count 0

# Scale back up
aws ecs update-service --cluster cow-lameness-production-cluster \
    --service ml-pipeline --desired-count 1
```

### Force Redeploy (after image update)

```bash
aws ecs update-service --cluster cow-lameness-production-cluster \
    --service admin-backend --force-new-deployment
```

### Shutdown Everything (minimize costs)

```bash
# Stop all ECS services
for svc in $(aws ecs list-services --cluster cow-lameness-production-cluster \
    --query 'serviceArns[*]' --output text); do
  aws ecs update-service --cluster cow-lameness-production-cluster \
      --service $(echo $svc | rev | cut -d/ -f1 | rev) --desired-count 0
done

# Stop RDS
aws rds stop-db-instance --db-instance-identifier cow-lameness-production-postgres
```

## ECR Repositories

| Repository | Image | Built By |
|------------|-------|----------|
| `admin-backend` | FastAPI backend | `build-gpu-images.yml` |
| `admin-frontend` | React frontend | `build-gpu-images.yml` |
| `video-ingestion` | Video upload service | `build-gpu-images.yml` |
| `gpu-inference` | Unified GPU model server | `build-sagemaker-images.yml` |
| `sagemaker-bridge` | NATS → SageMaker bridge | `build-sagemaker-images.yml` |
| `yolo-pipeline` | YOLO detection | `build-gpu-images.yml` |
| ... | (all pipeline services) | `build-gpu-images.yml` |

## CI/CD Workflows

| Workflow | Trigger | Builds |
|----------|---------|--------|
| `build-gpu-images.yml` | Push to `services/*/Dockerfile.gpu` | 8 GPU pipeline images |
| `build-sagemaker-images.yml` | Push to `sagemaker/`, `services/sagemaker-bridge/`, `shared/` | gpu-inference + sagemaker-bridge |

Both workflows push to ECR in `us-west-2` with `latest` + commit SHA tags.

## IAM Requirements

The Terraform user needs these AWS managed policies:

- AmazonEC2FullAccess
- AmazonECS_FullAccess
- AmazonRDSFullAccess
- AmazonS3FullAccess
- AmazonVPCFullAccess
- ElasticLoadBalancingFullAccess
- AmazonEC2ContainerRegistryFullAccess
- AmazonElasticFileSystemFullAccess
- IAMFullAccess
- SecretsManagerReadWrite
- CloudFrontFullAccess
- AmazonSageMakerFullAccess

Plus inline policies for: CloudWatch Logs, Service Discovery, Route53, ACM, Application Auto Scaling.
