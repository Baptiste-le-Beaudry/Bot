# terraform/variables.tf
# Variables de configuration pour l'infrastructure du robot de trading IA

# Variables générales
variable "project_name" {
  description = "Nom du projet utilisé pour nommer les ressources"
  type        = string
  default     = "ai-trading-bot"
}

variable "environment" {
  description = "Environnement de déploiement (dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "L'environnement doit être dev, staging ou prod."
  }
}

variable "aws_region" {
  description = "Région AWS pour le déploiement"
  type        = string
  default     = "us-east-1" # Proximité avec les principales bourses US
}

variable "terraform_state_bucket" {
  description = "Bucket S3 pour stocker l'état Terraform"
  type        = string
  default     = "ai-trading-bot-terraform-state"
}

# Variables réseau
variable "vpc_cidr" {
  description = "CIDR block pour le VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_vpn_gateway" {
  description = "Activer la passerelle VPN pour connexions sécurisées"
  type        = bool
  default     = true
}

# Variables EKS
variable "eks_cluster_version" {
  description = "Version de Kubernetes pour le cluster EKS"
  type        = string
  default     = "1.28"
}

variable "trading_nodes_count" {
  description = "Nombre de nœuds dédiés au trading"
  type        = number
  default     = 3
}

variable "trading_nodes_min" {
  description = "Nombre minimum de nœuds de trading"
  type        = number
  default     = 2
}

variable "trading_nodes_max" {
  description = "Nombre maximum de nœuds de trading"
  type        = number
  default     = 10
}

variable "ml_nodes_count" {
  description = "Nombre de nœuds GPU pour l'entraînement ML"
  type        = number
  default     = 2
}

variable "ml_nodes_min" {
  description = "Nombre minimum de nœuds ML"
  type        = number
  default     = 1
}

variable "ml_nodes_max" {
  description = "Nombre maximum de nœuds ML"
  type        = number
  default     = 5
}

# Variables base de données
variable "db_username" {
  description = "Nom d'utilisateur pour la base de données RDS"
  type        = string
  default     = "tradingbot"
  sensitive   = true
}

variable "db_password" {
  description = "Mot de passe pour la base de données RDS"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.db_password) >= 16
    error_message = "Le mot de passe doit contenir au moins 16 caractères."
  }
}

variable "db_instance_class" {
  description = "Type d'instance pour RDS"
  type        = string
  default     = "db.r6i.8xlarge"
}

variable "db_allocated_storage" {
  description = "Stockage alloué pour RDS en GB"
  type        = number
  default     = 1000
}

variable "db_iops" {
  description = "IOPS provisionnés pour RDS"
  type        = number
  default     = 64000
}

# Variables cache Redis
variable "redis_node_type" {
  description = "Type de nœud pour ElastiCache Redis"
  type        = string
  default     = "cache.r6g.8xlarge"
}

variable "redis_num_cache_nodes" {
  description = "Nombre de nœuds dans le cluster Redis"
  type        = number
  default     = 3
}

# Variables Kafka (MSK)
variable "kafka_instance_type" {
  description = "Type d'instance pour les brokers Kafka"
  type        = string
  default     = "kafka.m5.4xlarge"
}

variable "kafka_number_of_brokers" {
  description = "Nombre de brokers Kafka"
  type        = number
  default     = 3
}

variable "kafka_volume_size" {
  description = "Taille du volume EBS par broker en GB"
  type        = number
  default     = 1000
}

# Variables de monitoring
variable "log_retention_days" {
  description = "Durée de rétention des logs CloudWatch en jours"
  type        = number
  default     = 30
}

variable "enable_enhanced_monitoring" {
  description = "Activer le monitoring amélioré pour RDS"
  type        = bool
  default     = true
}

# Variables de sécurité
variable "enable_encryption_at_rest" {
  description = "Activer le chiffrement au repos pour toutes les ressources"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "Blocs CIDR autorisés pour l'accès externe"
  type        = list(string)
  default     = []
}

variable "enable_deletion_protection" {
  description = "Activer la protection contre la suppression pour les ressources critiques"
  type        = bool
  default     = true
}

# Variables de haute disponibilité
variable "multi_az_deployment" {
  description = "Activer le déploiement multi-AZ pour haute disponibilité"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Période de rétention des sauvegardes en jours"
  type        = number
  default     = 7
}

# Variables de performance
variable "enable_performance_insights" {
  description = "Activer Performance Insights pour RDS"
  type        = bool
  default     = true
}

variable "performance_insights_retention" {
  description = "Période de rétention Performance Insights en jours"
  type        = number
  default     = 7
}

# Variables de coûts
variable "use_spot_instances" {
  description = "Utiliser des instances Spot pour les nœuds non critiques"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Prix maximum pour les instances Spot"
  type        = string
  default     = ""
}

# Tags communs
variable "common_tags" {
  description = "Tags communs appliqués à toutes les ressources"
  type        = map(string)
  default = {
    Project     = "AI-Trading-Bot"
    ManagedBy   = "Terraform"
    CostCenter  = "Trading-Operations"
  }
}

# Variables spécifiques au trading
variable "enable_colocation" {
  description = "Activer les services de colocation pour latence ultra-faible"
  type        = bool
  default     = true
}

variable "exchange_endpoints" {
  description = "Endpoints des bourses pour connexions directes"
  type        = map(string)
  default = {
    binance  = "api.binance.com"
    coinbase = "api.exchange.coinbase.com"
    kraken   = "api.kraken.com"
  }
}

variable "max_orders_per_second" {
  description = "Limite maximale d'ordres par seconde"
  type        = number
  default     = 1000
}

variable "trading_pairs" {
  description = "Paires de trading actives"
  type        = list(string)
  default = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT"
  ]
}

# Variables ML/AI
variable "ml_model_s3_prefix" {
  description = "Préfixe S3 pour stocker les modèles ML"
  type        = string
  default     = "models/"
}

variable "training_schedule" {
  description = "Expression cron pour l'entraînement périodique des modèles"
  type        = string
  default     = "0 2 * * 0" # Tous les dimanches à 2h00
}

variable "gpu_instance_types" {
  description = "Types d'instances GPU pour l'entraînement ML"
  type        = list(string)
  default = [
    "p4d.24xlarge",
    "p3.16xlarge"
  ]
}

# Variables de monitoring et alertes
variable "alert_email" {
  description = "Email pour les alertes critiques"
  type        = string
  default     = ""
}

variable "slack_webhook_url" {
  description = "URL webhook Slack pour les notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_anomaly_detection" {
  description = "Activer la détection d'anomalies CloudWatch"
  type        = bool
  default     = true
}

# Variables de conformité
variable "enable_audit_logging" {
  description = "Activer les logs d'audit pour conformité"
  type        = bool
  default     = true
}

variable "compliance_standards" {
  description = "Standards de conformité à respecter"
  type        = list(string)
  default = [
    "SOC2",
    "ISO27001"
  ]
}