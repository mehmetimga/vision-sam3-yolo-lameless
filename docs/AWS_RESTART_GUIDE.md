# AWS Infrastructure Restart Guide

## Current State (Shutdown)
- ECS Services: Scaled to 0
- RDS Database: Stopped
- NAT Gateway: Deleted
- GPU: SageMaker endpoint (scales to 0 automatically)

**Estimated cost while shutdown: ~$25-30/month** (ALB + S3 + EFS only)

---

## Quick Restart Commands

### Step 1: Recreate NAT Gateway (Required for ECS)
```bash
cd /Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/terraform
terraform apply -target=module.networking.aws_nat_gateway.main -target=module.networking.aws_eip.nat -auto-approve
```

### Step 2: Start RDS Database
```bash
aws rds start-db-instance --db-instance-identifier cow-lameness-production-postgres --region us-west-2
# Wait 5-10 minutes for RDS to be available
aws rds wait db-instance-available --db-instance-identifier cow-lameness-production-postgres --region us-west-2
```

### Step 3: Scale Up ECS Services
```bash
for service in nats qdrant admin-backend admin-frontend video-ingestion video-preprocessing clip-curation tracking-service ml-pipeline fusion-service annotation-renderer sagemaker-bridge; do
  aws ecs update-service --cluster cow-lameness-production-cluster --service $service --desired-count 1 --region us-west-2
done
```

> **Note:** `sagemaker-bridge` is included above. It orchestrates GPU inference via SageMaker.
> The SageMaker endpoint itself scales to zero automatically — no manual start needed.

---

## Full Restart Script

Save this as `restart-aws.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== Restarting AWS Infrastructure ==="

# Step 1: NAT Gateway
echo "Step 1: Creating NAT Gateway..."
cd /Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/terraform
terraform apply -target=module.networking.aws_nat_gateway.main -target=module.networking.aws_eip.nat -auto-approve

# Step 2: RDS
echo "Step 2: Starting RDS..."
aws rds start-db-instance --db-instance-identifier cow-lameness-production-postgres --region us-west-2 2>/dev/null || echo "RDS may already be running"
echo "Waiting for RDS to be available (5-10 min)..."
aws rds wait db-instance-available --db-instance-identifier cow-lameness-production-postgres --region us-west-2
echo "RDS is ready!"

# Step 3: ECS Services
echo "Step 3: Scaling up ECS services..."
for service in nats qdrant admin-backend admin-frontend video-ingestion video-preprocessing clip-curation tracking-service ml-pipeline fusion-service annotation-renderer sagemaker-bridge; do
  aws ecs update-service --cluster cow-lameness-production-cluster --service $service --desired-count 1 --region us-west-2 --query "service.serviceName" --output text
done

echo ""
echo "=== Restart Complete ==="
echo "Application URL: https://cowhealth.ai"
echo "ALB Direct: https://cow-lameness-production-alb-292250301.us-west-2.elb.amazonaws.com"
echo ""
echo "SageMaker GPU endpoint scales up automatically when videos are processed."
echo "First video after restart will have ~5-10 min cold start."
```

---

## GPU Inference

GPU inference uses **SageMaker async endpoints** (pay-per-use, scale-to-zero).

- **No manual GPU start needed.** The `sagemaker-bridge` ECS service handles everything.
- When a video is uploaded, the bridge calls SageMaker, which auto-scales a GPU instance.
- After 10 minutes of idle, the GPU instance scales back to zero.
- First video after idle: ~5-10 minute cold start.

### If using EC2 GPU mode instead

To switch back to always-on EC2 GPU, edit `terraform/terraform.tfvars`:
```hcl
gpu_enabled       = true
sagemaker_enabled = false
```
Then run `terraform apply` and start the GPU worker:
```bash
aws autoscaling set-desired-capacity --auto-scaling-group-name cow-lameness-production-gpu-worker-asg --desired-capacity 1 --region us-west-2
```

---

## Before Restarting - Verify Images

Images are built via GitHub Actions and pushed to ECR automatically.

**ECR Registry:** `703582588105.dkr.ecr.us-west-2.amazonaws.com`

Key images:
- `gpu-inference:latest` — SageMaker GPU container (YOLO, SAM3, DINOv3, T-LEAP)
- `sagemaker-bridge:latest` — NATS → SageMaker orchestrator
- `admin-backend`, `admin-frontend`, and all pipeline services

To rebuild SageMaker images:
```bash
gh workflow run build-sagemaker-images.yml
```

---

## Important Notes

1. **RDS Auto-Start**: AWS automatically restarts stopped RDS instances after 7 days. If you need longer shutdown, stop it again.

2. **NAT Gateway**: Must be recreated before ECS services can start (they need internet access for ECR image pulls).

3. **Terraform State**: All infrastructure is managed by Terraform. The state file tracks deleted resources.

4. **SageMaker Cold Start**: The first video processed after the endpoint has been idle will take 5-10 minutes. Subsequent videos process in ~15 seconds.

5. **DNS**: The domain `cowhealth.ai` is managed in Route53 and points to the ALB. If the ALB is recreated, update the Route53 A records.

---

## Verification Commands

```bash
# Check all services are running
aws ecs describe-services --cluster cow-lameness-production-cluster \
    --services admin-frontend admin-backend nats sagemaker-bridge \
    --query 'services[*].{name:serviceName,running:runningCount}' \
    --region us-west-2

# Check SageMaker endpoint
aws sagemaker describe-endpoint \
    --endpoint-name cow-lameness-production-gpu-inference \
    --query 'EndpointStatus' --region us-west-2

# Check RDS
aws rds describe-db-instances \
    --db-instance-identifier cow-lameness-production-postgres \
    --query 'DBInstances[0].DBInstanceStatus' --region us-west-2

# Test frontend
curl -sI https://cowhealth.ai | head -3
```

---

## Cost Summary

| State | Monthly Cost |
|-------|-------------|
| Full Shutdown | ~$25-30 |
| ECS Only (no GPU) | ~$240 |
| SageMaker (light use, 2h GPU/day) | ~$285-330 |
| EC2 GPU Spot (always-on) | ~$360 |
| EC2 GPU On-Demand (always-on) | ~$980 |
