#!/bin/bash
# Build and push SageMaker inference images to ECR
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-2}"
ECR_REGISTRY="${ECR_REGISTRY:-703582588105.dkr.ecr.${AWS_REGION}.amazonaws.com}"
TAG="${TAG:-latest}"

echo "============================================"
echo "  Building SageMaker Images"
echo "  Registry: ${ECR_REGISTRY}"
echo "  Tag: ${TAG}"
echo "============================================"

cd "$(dirname "$0")/.."

echo ""
echo "--- Logging into ECR ---"
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo ""
echo "--- Creating ECR repositories (if needed) ---"
for REPO in gpu-inference sagemaker-bridge; do
    aws ecr describe-repositories --repository-names "${REPO}" --region "${AWS_REGION}" 2>/dev/null || \
        aws ecr create-repository --repository-name "${REPO}" --region "${AWS_REGION}"
done

echo ""
echo "--- Building unified GPU inference image ---"
docker build -f sagemaker/Dockerfile -t "${ECR_REGISTRY}/gpu-inference:${TAG}" .
docker push "${ECR_REGISTRY}/gpu-inference:${TAG}"
echo "Pushed: ${ECR_REGISTRY}/gpu-inference:${TAG}"

echo ""
echo "--- Building SageMaker bridge image ---"
docker build -f services/sagemaker-bridge/Dockerfile -t "${ECR_REGISTRY}/sagemaker-bridge:${TAG}" .
docker push "${ECR_REGISTRY}/sagemaker-bridge:${TAG}"
echo "Pushed: ${ECR_REGISTRY}/sagemaker-bridge:${TAG}"

echo ""
echo "============================================"
echo "  SageMaker images built and pushed!"
echo ""
echo "  Images:"
echo "    ${ECR_REGISTRY}/gpu-inference:${TAG}"
echo "    ${ECR_REGISTRY}/sagemaker-bridge:${TAG}"
echo "============================================"
