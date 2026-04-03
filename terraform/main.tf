# AWS Terraform Infrastructure for Cow Lameness Detection Platform
# Main configuration file

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state management
  # backend "s3" {
  #   bucket         = "cow-lameness-terraform-state"
  #   key            = "terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "cow-lameness-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Provider without default_tags for SageMaker (IAM user lacks sagemaker:AddTags)
provider "aws" {
  alias  = "no_default_tags"
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local variables
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  azs         = slice(data.aws_availability_zones.available.names, 0, 2)

  # ECS Services configuration
  ecs_services = {
    "admin-backend" = {
      cpu    = 2048
      memory = 4096
      port   = 8000
    }
    "admin-frontend" = {
      cpu    = 256
      memory = 512
      port   = 3000
    }
    "nats" = {
      cpu    = 256
      memory = 512
      port   = 4222
    }
    "qdrant" = {
      cpu    = 1024
      memory = 2048
      port   = 6333
    }
    "video-ingestion" = {
      cpu    = 512
      memory = 1024
      port   = 8001
    }
    "video-preprocessing" = {
      cpu    = 1024
      memory = 2048
      port   = 8002
    }
    "clip-curation" = {
      cpu    = 512
      memory = 1024
      port   = 8003
    }
    "tracking-service" = {
      cpu    = 512
      memory = 1024
      port   = 8004
    }
    "ml-pipeline" = {
      cpu    = 1024
      memory = 2048
      port   = 8005
    }
    "fusion-service" = {
      cpu    = 512
      memory = 1024
      port   = 8006
    }
    "annotation-renderer" = {
      cpu    = 1024
      memory = 2048
      port   = 8000
    }
  }

  # GPU services (run on EC2 when gpu_enabled, ignored when using SageMaker)
  gpu_services = [
    "yolo-pipeline",
    "sam3-pipeline",
    "dinov3-pipeline",
    "tleap-pipeline",
    "tcn-pipeline",
    "transformer-pipeline",
    "gnn-pipeline",
    "graph-transformer-pipeline"
  ]

  # SageMaker bridge service (added to ECS when SageMaker is enabled)
  sagemaker_ecs_services = var.sagemaker_enabled ? {
    "sagemaker-bridge" = {
      cpu    = 512
      memory = 1024
      port   = 8007
    }
  } : {}

  all_ecs_services = merge(local.ecs_services, local.sagemaker_ecs_services)
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  azs         = local.azs
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  name_prefix        = local.name_prefix
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  efs_security_group = module.networking.efs_security_group_id
  enable_cloudfront  = var.enable_cloudfront
}

# Database Module
module "database" {
  source = "./modules/database"

  name_prefix          = local.name_prefix
  vpc_id               = module.networking.vpc_id
  database_subnet_ids  = module.networking.database_subnet_ids
  db_security_group_id = module.networking.db_security_group_id
  db_password          = var.db_password
}

# Secrets Module
module "secrets" {
  source = "./modules/secrets"

  name_prefix = local.name_prefix
  db_password = var.db_password
  jwt_secret  = var.jwt_secret
  db_endpoint = module.database.db_endpoint
  db_name     = module.database.db_name
}

# ECS Module
module "ecs" {
  source = "./modules/ecs"

  name_prefix             = local.name_prefix
  vpc_id                  = module.networking.vpc_id
  private_subnet_ids      = module.networking.private_subnet_ids
  ecs_security_group_id   = module.networking.ecs_security_group_id
  ecs_services            = local.all_ecs_services
  ecr_registry            = var.ecr_registry
  efs_file_system_id      = module.storage.efs_file_system_id
  efs_access_point_id     = module.storage.efs_access_point_id
  secrets_arn             = module.secrets.secrets_arn
  alb_target_group_arn    = module.load_balancer.backend_target_group_arn
  frontend_target_group_arn = module.load_balancer.frontend_target_group_arn
  videos_bucket_name      = module.storage.videos_bucket_name
  cloudfront_domain       = module.storage.cloudfront_domain_name
  image_tag               = var.image_tag
  sagemaker_endpoint_name = var.sagemaker_enabled ? module.sagemaker[0].endpoint_name : ""
  sagemaker_io_bucket     = var.sagemaker_enabled ? module.sagemaker[0].sagemaker_io_bucket : ""
}

# Load Balancer Module
module "load_balancer" {
  source = "./modules/load_balancer"

  name_prefix       = local.name_prefix
  vpc_id            = module.networking.vpc_id
  public_subnet_ids = module.networking.public_subnet_ids
  alb_security_group_id = module.networking.alb_security_group_id
  certificate_arn   = var.certificate_arn
}

# GPU Worker Module (EC2-based, used when NOT using SageMaker)
module "gpu_worker" {
  source = "./modules/gpu_worker"

  name_prefix            = local.name_prefix
  vpc_id                 = module.networking.vpc_id
  private_subnet_ids     = module.networking.private_subnet_ids
  gpu_security_group_id  = module.networking.gpu_security_group_id
  gpu_enabled            = var.gpu_enabled && !var.sagemaker_enabled
  gpu_instance_type      = var.gpu_instance_type
  use_spot_instances     = var.use_spot_instances
  efs_file_system_id     = module.storage.efs_file_system_id
  ecr_registry           = var.ecr_registry
  nats_endpoint          = "nats.${local.name_prefix}.local:4222"
  gpu_services           = local.gpu_services
}

# SageMaker Module (pay-per-use GPU inference with scale-to-zero)
module "sagemaker" {
  source = "./modules/sagemaker"
  count  = var.sagemaker_enabled ? 1 : 0

  providers = {
    aws = aws.no_default_tags
  }

  name_prefix             = local.name_prefix
  ecr_registry            = var.ecr_registry
  private_subnet_ids      = module.networking.private_subnet_ids
  security_group_id       = module.networking.gpu_security_group_id
  videos_bucket_name      = module.storage.videos_bucket_name
  sagemaker_instance_type = var.sagemaker_instance_type
  sagemaker_max_instances = var.sagemaker_max_instances
}
