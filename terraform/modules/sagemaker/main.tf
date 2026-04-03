# SageMaker Module - Async inference endpoints with scale-to-zero
# Replaces always-on GPU EC2 with pay-per-use GPU inference

# IAM Role for SageMaker execution
resource "aws_iam_role" "sagemaker" {
  name = "${var.name_prefix}-sagemaker"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_custom" {
  name = "${var.name_prefix}-sagemaker-custom"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_io.arn,
          "${aws_s3_bucket.sagemaker_io.arn}/*",
          "arn:aws:s3:::${var.videos_bucket_name}",
          "arn:aws:s3:::${var.videos_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:CreateLogGroup"
        ]
        Resource = "*"
      }
    ]
  })
}

# S3 bucket for async inference input/output
resource "aws_s3_bucket" "sagemaker_io" {
  bucket        = "${var.name_prefix}-sagemaker-io"
  force_destroy = true
}

resource "aws_s3_bucket_lifecycle_configuration" "sagemaker_io" {
  bucket = aws_s3_bucket.sagemaker_io.id

  rule {
    id     = "cleanup-input"
    status = "Enabled"
    filter { prefix = "input/" }
    expiration { days = 3 }
  }

  rule {
    id     = "cleanup-output"
    status = "Enabled"
    filter { prefix = "output/" }
    expiration { days = 7 }
  }

  rule {
    id     = "cleanup-video-cache"
    status = "Enabled"
    filter { prefix = "videos/" }
    expiration { days = 1 }
  }
}

# Unified GPU inference model (all vision pipelines in one container)
resource "aws_sagemaker_model" "gpu_inference" {
  name               = "${var.name_prefix}-gpu-inference"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image = "${var.ecr_registry}/gpu-inference:latest"
    mode  = "SingleModel"
    environment = {
      SAGEMAKER_MODE  = "true"
      S3_BUCKET       = aws_s3_bucket.sagemaker_io.id
      VIDEOS_BUCKET   = var.videos_bucket_name
      MODEL_CACHE_DIR = "/opt/ml/model"
    }
  }

  vpc_config {
    security_group_ids = [var.security_group_id]
    subnets            = var.private_subnet_ids
  }
}

# Async inference endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "gpu_inference" {
  name = "${var.name_prefix}-gpu-inference"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.gpu_inference.name
    instance_type          = var.sagemaker_instance_type
    initial_instance_count = 1
  }

  async_inference_config {
    output_config {
      s3_output_path = "s3://${aws_s3_bucket.sagemaker_io.id}/output"
    }

    client_config {
      max_concurrent_invocations_per_instance = 2
    }
  }
}

# SageMaker endpoint
resource "aws_sagemaker_endpoint" "gpu_inference" {
  name                 = "${var.name_prefix}-gpu-inference"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.gpu_inference.name
}

# Auto-scaling: scale to 0 when idle, scale to 1 when requests arrive
resource "aws_appautoscaling_target" "gpu_inference" {
  max_capacity       = var.sagemaker_max_instances
  min_capacity       = 0
  resource_id        = "endpoint/${aws_sagemaker_endpoint.gpu_inference.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
}

resource "aws_appautoscaling_policy" "scale_on_backlog" {
  name               = "${var.name_prefix}-gpu-scale-on-backlog"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.gpu_inference.resource_id
  scalable_dimension = aws_appautoscaling_target.gpu_inference.scalable_dimension
  service_namespace  = aws_appautoscaling_target.gpu_inference.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 1.0

    customized_metric_specification {
      metric_name = "ApproximateBacklogSizePerInstance"
      namespace   = "AWS/SageMaker"
      statistic   = "Average"

      dimensions {
        name  = "EndpointName"
        value = aws_sagemaker_endpoint.gpu_inference.name
      }
    }

    scale_in_cooldown  = 600
    scale_out_cooldown = 60
  }
}

# CloudWatch log group
resource "aws_cloudwatch_log_group" "sagemaker" {
  name              = "/sagemaker/${var.name_prefix}"
  retention_in_days = 30
}
