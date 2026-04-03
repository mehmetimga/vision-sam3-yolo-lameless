variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "ecr_registry" {
  description = "ECR registry URL"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for SageMaker VPC config"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group for SageMaker endpoints"
  type        = string
}

variable "videos_bucket_name" {
  description = "S3 bucket name for videos"
  type        = string
}

variable "sagemaker_instance_type" {
  description = "Instance type for SageMaker inference endpoints"
  type        = string
  default     = "ml.g4dn.xlarge"
}

variable "sagemaker_max_instances" {
  description = "Maximum number of instances for auto-scaling"
  type        = number
  default     = 1
}
