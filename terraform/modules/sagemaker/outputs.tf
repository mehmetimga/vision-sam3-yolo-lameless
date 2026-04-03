output "endpoint_name" {
  description = "SageMaker endpoint name for GPU inference"
  value       = aws_sagemaker_endpoint.gpu_inference.name
}

output "sagemaker_io_bucket" {
  description = "S3 bucket for SageMaker async I/O"
  value       = aws_s3_bucket.sagemaker_io.id
}

output "sns_topic_arn" {
  description = "SNS topic ARN for inference notifications"
  value       = aws_sns_topic.inference_notifications.arn
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker.arn
}
