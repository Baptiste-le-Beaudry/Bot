# Docker Deployment Guide for AI Trading Robot

This directory contains Docker configuration for containerizing and running the AI Trading Robot.

## 📁 Files Overview

### Dockerfile
- **Multi-stage build** for optimized image size
- **Production stage**: Minimal runtime with security hardening
- **GPU stage**: CUDA support for ML acceleration
- **Development stage**: Includes debugging tools
- **Security**: Non-root user, minimal attack surface
- **Health checks**: Built-in liveness probes

### docker-compose.yml
Complete local environment with:
- **Trading Bot**: Main application container
- **PostgreSQL + TimescaleDB**: Time-series database
- **Redis**: High-performance cache
- **Kafka + Zookeeper**: Event streaming
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Elasticsearch + Kibana**: Log management
- **AlertManager**: Alert routing

## 🚀 Quick Start

### 1. Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM available
- 20GB+ disk space

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

### 3. Build and Run

#### Development Mode
```bash
# Build development image
docker-compose build --build-arg BUILD_TARGET=development

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot
```

#### Production Mode
```bash
# Build production image
docker-compose build

# Start services
docker-compose up -d

# Scale trading bot
docker-compose up -d --scale trading-bot=3
```

#### GPU Mode (for ML acceleration)
```bash
# Build GPU image
docker-compose build --build-arg BUILD_TARGET=gpu

# Run with GPU support
docker-compose run --gpus all trading-bot
```

## 📊 Service URLs

After starting, services are available at:

- **Trading Bot API**: http://localhost:8080
- **WebSocket**: ws://localhost:8081
- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboards**: http://localhost:3000 (admin/admin123)
- **Kibana Logs**: http://localhost:5601
- **pgAdmin**: http://localhost:5050 (admin@trading-bot.local/admin123)
- **RedisInsight**: http://localhost:8001

## 🔧 Configuration

### Environment Variables
Key variables in `.env`:
```bash
TRADING_ENV=development      # or production
BUILD_TARGET=production      # or development, gpu
BINANCE_TESTNET=true        # Use testnet for safety
LOG_LEVEL=INFO              # DEBUG for troubleshooting
```

### Resource Limits
Adjust in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Persistent Data
Data is stored in Docker volumes:
- `trading-data`: Application data
- `trading-models`: ML models
- `postgres-data`: Database
- `redis-data`: Cache

## 🛠️ Common Operations

### Database Management
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U trader -d trading

# Backup database
docker-compose exec postgres pg_dump -U trader trading > backup.sql

# Restore database
docker-compose exec postgres psql -U trader trading < backup.sql
```

### Redis Management
```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli -a redis123

# Monitor Redis commands
docker-compose exec redis redis-cli -a redis123 monitor

# Flush cache
docker-compose exec redis redis-cli -a redis123 FLUSHALL
```

### Application Management
```bash
# Execute commands in container
docker-compose exec trading-bot python -m scripts.train_models

# Access Python shell
docker-compose exec trading-bot python

# Run tests
docker-compose exec trading-bot pytest tests/
```

### Monitoring
```bash
# View real-time metrics
curl http://localhost:9090/metrics

# Check health status
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready

# View container stats
docker stats
```

## 📈 Performance Tuning

### Docker Settings
1. **Increase Docker memory** (Docker Desktop):
   - Settings → Resources → Memory: 8GB+

2. **Enable BuildKit** for faster builds:
   ```bash
   export DOCKER_BUILDKIT=1
   docker-compose build
   ```

3. **Use .dockerignore** to exclude unnecessary files

### Application Optimization
1. **Connection Pooling**:
   ```yaml
   DB_POOL_SIZE: "20"
   REDIS_POOL_SIZE: "50"
   ```

2. **Worker Processes**:
   ```yaml
   WORKER_PROCESSES: "4"  # Match CPU cores
   ```

3. **Memory Settings**:
   ```yaml
   PYTHONMALLOC: "malloc"  # Better memory management
   ```

## 🔒 Security Best Practices

1. **Never commit secrets**:
   - Use `.env` files (git-ignored)
   - Use Docker secrets in production

2. **Network isolation**:
   - Services communicate via internal network
   - Only expose necessary ports

3. **Read-only filesystem**:
   - Application runs with read-only root
   - Only specific directories are writable

4. **Non-root user**:
   - Runs as `trader` user (UID 1000)
   - Minimal privileges

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs trading-bot

# Inspect container
docker-compose ps
docker inspect trading-bot
```

### Database connection issues
```bash
# Test connection
docker-compose exec trading-bot pg_isready -h postgres -U trader

# Check network
docker network ls
docker network inspect docker_trading-network
```

### High memory usage
```bash
# Check memory stats
docker stats --no-stream

# Limit memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

### Performance issues
```bash
# Profile the application
docker-compose exec trading-bot python -m cProfile -o profile.stats core.engine

# Monitor resource usage
docker-compose exec trading-bot htop
```

## 🔄 Backup and Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U trader trading > $BACKUP_DIR/postgres.sql

# Backup Redis
docker-compose exec -T redis redis-cli -a redis123 --rdb $BACKUP_DIR/redis.rdb

# Backup volumes
docker run --rm -v trading-data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/data.tar.gz /data
docker run --rm -v trading-models:/models -v $BACKUP_DIR:/backup alpine tar czf /backup/models.tar.gz /models
```

### Restore Process
```bash
# Stop services
docker-compose stop

# Restore database
docker-compose exec -T postgres psql -U trader trading < backup/postgres.sql

# Restore Redis
docker-compose exec -T redis redis-cli -a redis123 --rdb backup/redis.rdb

# Restore volumes
docker run --rm -v trading-data:/data -v ./backup:/backup alpine tar xzf /backup/data.tar.gz
docker run --rm -v trading-models:/models -v ./backup:/backup alpine tar xzf /backup/models.tar.gz

# Restart services
docker-compose start
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Best Practices for Python Docker Images](https://pythonspeed.com/docker/)
- [Container Security Guidelines](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

## 🆘 Support

For issues:
1. Check container logs: `docker-compose logs`
2. Verify environment variables: `docker-compose config`
3. Test health endpoints
4. Review resource usage

---

**Note**: This setup is optimized for local development and testing. For production deployment, consider using Kubernetes or cloud-native container services.