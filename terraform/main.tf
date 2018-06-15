# Terraform Configuration for Big Data ML Pipeline
# Deploys Spark cluster on AWS EMR with MLflow tracking

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for Data Lake
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${var.environment}"
  
  tags = {
    Name        = "ML Pipeline Data Lake"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket for MLflow Artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-mlflow-artifacts-${var.environment}"
  
  tags = {
    Name        = "MLflow Artifacts"
    Environment = var.environment
  }
}

# S3 Bucket for EMR Logs
resource "aws_s3_bucket" "emr_logs" {
  bucket = "${var.project_name}-emr-logs-${var.environment}"
  
  tags = {
    Name        = "EMR Logs"
    Environment = var.environment
  }
}

# IAM Role for EMR
resource "aws_iam_role" "emr_service_role" {
  name = "${var.project_name}-emr-service-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_service_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

# IAM Role for EMR EC2 Instances
resource "aws_iam_role" "emr_ec2_role" {
  name = "${var.project_name}-emr-ec2-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_ec2_policy" {
  role       = aws_iam_role.emr_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

resource "aws_iam_instance_profile" "emr_ec2_profile" {
  name = "${var.project_name}-emr-ec2-profile"
  role = aws_iam_role.emr_ec2_role.name
}

# Security Group for EMR
resource "aws_security_group" "emr_master" {
  name        = "${var.project_name}-emr-master-sg"
  description = "Security group for EMR master node"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  
  ingress {
    from_port   = 8088
    to_port     = 8088
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  
  ingress {
    from_port   = 4040
    to_port     = 4040
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "EMR Master Security Group"
  }
}

# EMR Cluster
resource "aws_emr_cluster" "ml_pipeline" {
  name          = "${var.project_name}-ml-pipeline-${var.environment}"
  release_label = "emr-6.15.0"
  applications  = ["Spark", "Hadoop", "Hive", "Livy"]
  
  service_role = aws_iam_role.emr_service_role.arn
  
  ec2_attributes {
    instance_profile                  = aws_iam_instance_profile.emr_ec2_profile.arn
    emr_managed_master_security_group = aws_security_group.emr_master.id
    subnet_id                         = var.subnet_id
  }
  
  master_instance_group {
    instance_type  = var.master_instance_type
    instance_count = 1
    
    ebs_config {
      size                 = 100
      type                 = "gp3"
      volumes_per_instance = 1
    }
  }
  
  core_instance_group {
    instance_type  = var.core_instance_type
    instance_count = var.core_instance_count
    
    ebs_config {
      size                 = 100
      type                 = "gp3"
      volumes_per_instance = 1
    }
    
    autoscaling_policy = jsonencode({
      Constraints = {
        MinCapacity = var.core_instance_count
        MaxCapacity = var.max_core_instances
      }
      Rules = [
        {
          Name = "ScaleOutMemoryPercentage"
          Description = "Scale out if YARNMemoryAvailablePercentage is less than 15"
          Action = {
            SimpleScalingPolicyConfiguration = {
              AdjustmentType = "CHANGE_IN_CAPACITY"
              ScalingAdjustment = 1
              CoolDown = 300
            }
          }
          Trigger = {
            CloudWatchAlarmDefinition = {
              ComparisonOperator = "LESS_THAN"
              EvaluationPeriods = 1
              MetricName = "YARNMemoryAvailablePercentage"
              Namespace = "AWS/ElasticMapReduce"
              Period = 300
              Statistic = "AVERAGE"
              Threshold = 15.0
              Unit = "PERCENT"
            }
          }
        }
      ]
    })
  }
  
  configurations_json = jsonencode([
    {
      Classification = "spark"
      Properties = {
        "maximizeResourceAllocation" = "true"
      }
    },
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.sql.adaptive.enabled"                    = "true"
        "spark.sql.adaptive.coalescePartitions.enabled" = "true"
        "spark.serializer"                              = "org.apache.spark.serializer.KryoSerializer"
        "spark.dynamicAllocation.enabled"               = "true"
        "spark.shuffle.service.enabled"                 = "true"
      }
    }
  ])
  
  log_uri = "s3://${aws_s3_bucket.emr_logs.bucket}/logs/"
  
  tags = {
    Name        = "ML Pipeline EMR Cluster"
    Environment = var.environment
  }
}

# RDS for MLflow Backend Store
resource "aws_db_instance" "mlflow" {
  identifier        = "${var.project_name}-mlflow-db"
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  
  db_name  = "mlflow"
  username = var.db_username
  password = var.db_password
  
  skip_final_snapshot = true
  
  tags = {
    Name        = "MLflow Database"
    Environment = var.environment
  }
}

# ECS for MLflow Tracking Server
resource "aws_ecs_cluster" "mlflow" {
  name = "${var.project_name}-mlflow-cluster"
  
  tags = {
    Name        = "MLflow Cluster"
    Environment = var.environment
  }
}

resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${var.project_name}-mlflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  
  container_definitions = jsonencode([
    {
      name  = "mlflow"
      image = "ghcr.io/mlflow/mlflow:latest"
      
      command = [
        "mlflow",
        "server",
        "--backend-store-uri",
        "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow.endpoint}/mlflow",
        "--default-artifact-root",
        "s3://${aws_s3_bucket.mlflow_artifacts.bucket}/",
        "--host",
        "0.0.0.0"
      ]
      
      portMappings = [
        {
          containerPort = 5000
          protocol      = "tcp"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/mlflow"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "mlflow"
        }
      }
    }
  ])
}

# Outputs
output "emr_cluster_id" {
  description = "EMR Cluster ID"
  value       = aws_emr_cluster.ml_pipeline.id
}

output "emr_master_public_dns" {
  description = "EMR Master Public DNS"
  value       = aws_emr_cluster.ml_pipeline.master_public_dns
}

output "data_lake_bucket" {
  description = "S3 Data Lake Bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "mlflow_artifacts_bucket" {
  description = "MLflow Artifacts Bucket"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "mlflow_db_endpoint" {
  description = "MLflow Database Endpoint"
  value       = aws_db_instance.mlflow.endpoint
}
