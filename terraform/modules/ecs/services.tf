# ECS Services Configuration

# Service Discovery Services
resource "aws_service_discovery_service" "services" {
  for_each = var.ecs_services

  name = each.key

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id

    dns_records {
      ttl  = 10
      type = "A"
    }

    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Task Definitions
resource "aws_ecs_task_definition" "services" {
  for_each = var.ecs_services

  family                   = "${var.name_prefix}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = each.key
      image     = var.ecr_registry != "" ? "${var.ecr_registry}/${each.key}:latest" : "${each.key}:latest"
      essential = true

      portMappings = [
        {
          containerPort = each.value.port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "NATS_URL"
          value = "nats://nats.${var.name_prefix}.local:4222"
        },
        {
          name  = "QDRANT_URL"
          value = "http://qdrant.${var.name_prefix}.local:6333"
        },
        {
          name  = "SERVICE_NAME"
          value = each.key
        }
      ]

      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = "${var.secrets_arn}:DATABASE_URL::"
        },
        {
          name      = "JWT_SECRET"
          valueFrom = "${var.secrets_arn}:JWT_SECRET::"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "efs-data"
          containerPath = "/app/data"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = each.key
        }
      }

      healthCheck = each.key == "nats" ? null : {
        command     = ["CMD-SHELL", "curl -f http://localhost:${each.value.port}/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  volume {
    name = "efs-data"

    efs_volume_configuration {
      file_system_id     = var.efs_file_system_id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = var.efs_access_point_id
        iam             = "ENABLED"
      }
    }
  }

  tags = {
    Name = "${var.name_prefix}-${each.key}"
  }
}

# ECS Services
resource "aws_ecs_service" "services" {
  for_each = var.ecs_services

  name            = each.key
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.services[each.key].arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.services[each.key].arn
  }

  # Load balancer configuration for backend and frontend
  dynamic "load_balancer" {
    for_each = each.key == "admin-backend" ? [1] : []
    content {
      target_group_arn = var.alb_target_group_arn
      container_name   = each.key
      container_port   = each.value.port
    }
  }

  dynamic "load_balancer" {
    for_each = each.key == "admin-frontend" ? [1] : []
    content {
      target_group_arn = var.frontend_target_group_arn
      container_name   = each.key
      container_port   = each.value.port
    }
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  lifecycle {
    ignore_changes = [desired_count, task_definition]
  }

  tags = {
    Name = "${var.name_prefix}-${each.key}"
  }
}

# Data source for current region
data "aws_region" "current" {}
