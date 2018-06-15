# Terraform Variables

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-pipeline"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for EMR"
  type        = string
}

variable "allowed_cidr" {
  description = "Allowed CIDR block for access"
  type        = string
  default     = "0.0.0.0/0"
}

variable "master_instance_type" {
  description = "EMR master instance type"
  type        = string
  default     = "m5.xlarge"
}

variable "core_instance_type" {
  description = "EMR core instance type"
  type        = string
  default     = "m5.xlarge"
}

variable "core_instance_count" {
  description = "Number of core instances"
  type        = number
  default     = 2
}

variable "max_core_instances" {
  description = "Maximum number of core instances for autoscaling"
  type        = number
  default     = 10
}

variable "db_username" {
  description = "Database username for MLflow"
  type        = string
  default     = "mlflow"
  sensitive   = true
}

variable "db_password" {
  description = "Database password for MLflow"
  type        = string
  sensitive   = true
}
