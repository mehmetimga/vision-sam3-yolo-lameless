# ECS Module Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_subnet_ids" {
  description = "IDs of private subnets"
  type        = list(string)
}

variable "ecs_security_group_id" {
  description = "Security group ID for ECS services"
  type        = string
}

variable "ecs_services" {
  description = "Map of ECS services to create"
  type = map(object({
    cpu    = number
    memory = number
    port   = number
  }))
}

variable "ecr_registry" {
  description = "ECR registry URL"
  type        = string
  default     = ""
}

variable "efs_file_system_id" {
  description = "EFS file system ID"
  type        = string
}

variable "efs_access_point_id" {
  description = "EFS access point ID"
  type        = string
}

variable "secrets_arn" {
  description = "ARN of the secrets in Secrets Manager"
  type        = string
}

variable "alb_target_group_arn" {
  description = "ARN of the ALB target group for backend"
  type        = string
}

variable "frontend_target_group_arn" {
  description = "ARN of the ALB target group for frontend"
  type        = string
}
