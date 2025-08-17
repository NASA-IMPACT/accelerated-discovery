# Deployment Documentation

This directory contains deployment configurations and scripts for various components of the Accelerated Discovery framework.

## SearxNG AWS Deployment

Complete AWS ECS deployment setup for SearxNG scientific search engine.

### Quick Start

```bash
# 1. Build and push to ECR
./scripts/push_searxng_to_ecr.sh

# 2. Deploy to ECS
./scripts/deploy-searxng-ecs.sh

# 3. Get public URL
aws ecs list-tasks --cluster searxng-cluster --service searxng-service
```

### Files

- `ecs-task-definition.json` - ECS Fargate task configuration
- `docker-compose-ecr.yml` - Docker Compose with ECR image
- `nginx.conf` - Production nginx configuration with SSL

### Documentation

See [SearxNG AWS Deployment Guide](../docs/searxng-aws-deployment.md) for complete setup instructions.

## Cost Estimation

- **Development**: ~$0.023/hour (~$16.56/month)
- **Production with ALB/SSL**: ~$20-30/month

## Security Features

- Network isolation with security groups
- Container security with non-root user
- Optional SSL termination with nginx
- CloudWatch logging for monitoring