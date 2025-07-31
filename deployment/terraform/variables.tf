# terraform/main.tf
# Infrastructure principale pour le robot de trading IA haute fréquence

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  backend "s3" {
    bucket = var.terraform_state_bucket
    key    = "trading-bot/terraform.tfstate"
    region = var.aws_region
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC pour isolation réseau et faible latence
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = var.common_tags
}

# Cluster EKS pour orchestration des conteneurs
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups pour différents workloads
  eks_managed_node_groups = {
    # Nodes haute performance pour trading
    trading_nodes = {
      desired_capacity = var.trading_nodes_count
      max_capacity     = var.trading_nodes_max
      min_capacity     = var.trading_nodes_min
      
      instance_types = ["m5n.24xlarge"] # Optimisé réseau pour faible latence
      
      k8s_labels = {
        Environment = var.environment
        Workload    = "trading"
      }
      
      additional_tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${var.project_name}-eks" = "owned"
      }
    }
    
    # Nodes pour ML/DRL training
    ml_nodes = {
      desired_capacity = var.ml_nodes_count
      max_capacity     = var.ml_nodes_max
      min_capacity     = var.ml_nodes_min
      
      instance_types = ["p4d.24xlarge"] # GPU pour entraînement DRL
      
      k8s_labels = {
        Environment = var.environment
        Workload    = "ml-training"
      }
    }
    
    # Nodes standard pour services auxiliaires
    general_nodes = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3
      
      instance_types = ["m5.2xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Workload    = "general"
      }
    }
  }
  
  tags = var.common_tags
}

# RDS pour stockage des données de trading
resource "aws_db_instance" "trading_db" {
  identifier = "${var.project_name}-trading-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6i.8xlarge" # Haute performance I/O
  
  allocated_storage     = 1000
  storage_type         = "io1"
  iops                 = 64000
  storage_encrypted    = true
  
  db_name  = "trading"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
  
  tags = var.common_tags
}

# TimeStream pour données de marché temps réel
resource "aws_timestreamwrite_database" "market_data" {
  database_name = "${var.project_name}-market-data"
  
  tags = var.common_tags
}

resource "aws_timestreamwrite_table" "trades" {
  database_name = aws_timestreamwrite_database.market_data.database_name
  table_name    = "trades"
  
  retention_properties {
    memory_store_retention_period_in_hours = 24
    magnetic_store_retention_period_in_days = 365
  }
  
  tags = var.common_tags
}

resource "aws_timestreamwrite_table" "orderbook" {
  database_name = aws_timestreamwrite_database.market_data.database_name
  table_name    = "orderbook"
  
  retention_properties {
    memory_store_retention_period_in_hours = 12
    magnetic_store_retention_period_in_days = 30
  }
  
  tags = var.common_tags
}

# S3 pour stockage des modèles et données historiques
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.project_name}-ml-models-${var.environment}"
  
  tags = var.common_tags
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket" "market_data_archive" {
  bucket = "${var.project_name}-market-data-archive-${var.environment}"
  
  tags = var.common_tags
}

# Kinesis pour streaming de données en temps réel
resource "aws_kinesis_stream" "market_data_stream" {
  name = "${var.project_name}-market-data-stream"
  
  stream_mode_details {
    stream_mode = "ON_DEMAND"
  }
  
  tags = var.common_tags
}

# ElastiCache Redis pour cache haute performance
resource "aws_elasticache_replication_group" "trading_cache" {
  replication_group_id       = "${var.project_name}-cache"
  replication_group_description = "Redis cache for trading bot"
  
  engine               = "redis"
  engine_version       = "7.0"
  node_type           = "cache.r6g.8xlarge"
  number_cache_clusters = 3
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = var.common_tags
}

# MSK (Kafka) pour event streaming
resource "aws_msk_cluster" "trading_events" {
  cluster_name           = "${var.project_name}-events"
  kafka_version         = "3.5.1"
  number_of_broker_nodes = 3
  
  broker_node_group_info {
    instance_type   = "kafka.m5.4xlarge"
    client_subnets = module.vpc.private_subnets
    
    storage_info {
      ebs_storage_info {
        volume_size = 1000
        provisioned_throughput {
          enabled           = true
          volume_throughput = 1000
        }
      }
    }
    
    security_groups = [aws_security_group.msk.id]
  }
  
  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }
  
  tags = var.common_tags
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}

resource "aws_security_group" "msk" {
  name_prefix = "${var.project_name}-msk-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 9092
    to_port     = 9098
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}

# Subnet Groups
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets
  
  tags = var.common_tags
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
  
  tags = var.common_tags
}

# IAM Roles pour EKS
resource "aws_iam_role" "eks_node_group" {
  name = "${var.project_name}-eks-node-group"
  
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
  
  tags = var.common_tags
}

# ECR pour images Docker
resource "aws_ecr_repository" "trading_bot" {
  name = "${var.project_name}/trading-bot"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = var.common_tags
}

resource "aws_ecr_repository" "ml_trainer" {
  name = "${var.project_name}/ml-trainer"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = var.common_tags
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "trading_logs" {
  name              = "/aws/eks/${var.project_name}/trading"
  retention_in_days = 30
  
  tags = var.common_tags
}

# Outputs
output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.trading_db.endpoint
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.trading_cache.primary_endpoint_address
}

output "msk_bootstrap_brokers" {
  description = "MSK cluster bootstrap brokers"
  value       = aws_msk_cluster.trading_events.bootstrap_brokers
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}