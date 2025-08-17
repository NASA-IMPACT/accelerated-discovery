# SearxNG AWS Deployment Guide

Complete guide for deploying SearxNG to AWS ECS with public access.

## Overview

This document provides step-by-step instructions for deploying a custom SearxNG search engine to AWS ECS Fargate with public internet access. The deployment includes:

- Custom SearxNG configuration optimized for scientific research
- Docker containerization with ECR storage
- AWS ECS Fargate serverless deployment
- Public access with security groups
- CloudWatch logging

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed and running
- Access to AWS account with ECS, ECR, EC2, and CloudWatch permissions

## Architecture

```
Internet → Security Group (Port 8080) → ECS Fargate Task → SearxNG Container
                                      ↓
                                CloudWatch Logs
                                      ↓
                               ECR Image Repository
```

## Step 1: SearxNG Configuration

### Custom Settings File

Create `searxng/searxng_settings.yml` with scientific search optimization:

```yaml
use_default_settings: true

general:
  debug: false
  instance_name: "SearxNG scientific"
  privacypolicy_url: false
  donation_url: false
  contact_url: false
  enable_metrics: false

search:
  safe_search: 2
  autocomplete: "duckduckgo"
  default_lang: "en"
  default_category: "science"
  formats:
    - json
    - html

server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "akd_secret_key_12345"
  base_url: "http://localhost:8080/"
  image_proxy: false
  http_protocol_version: "1.1"
  method: "POST"
  limiter: false  # Disabled to avoid Valkey dependency
  default_http_headers:
    X-Content-Type-Options: nosniff
    X-XSS-Protection: 1; mode=block
    X-Download-Options: noopen
    X-Robots-Tag: noindex, nofollow
    Referrer-Policy: no-referrer
    Access-Control-Allow-Origin: "*"

# Enable scientific search engines
engines:
  - name: google scholar
    disabled: false
    timeout: 6.0
  - name: arxiv
    disabled: false
    timeout: 6.0
  - name: pubmed
    disabled: false
    timeout: 6.0
  - name: semantic scholar
    disabled: false
    timeout: 6.0
  - name: crossref
    disabled: false
    timeout: 6.0
  - name: doi
    disabled: false
    timeout: 6.0
  - name: openalex
    disabled: false
    timeout: 6.0
```

### Dockerfile

Create `searxng/Dockerfile`:

```dockerfile
FROM searxng/searxng:latest

# Copy custom configuration
COPY searxng_settings.yml /etc/searxng/settings.yml

# Set proper permissions
RUN chown -R searxng:searxng /etc/searxng/

# Expose port
EXPOSE 8080

# Use the default entrypoint from the base image
```

## Step 2: ECR Repository Setup

### Create ECR Repository

```bash
# Create repository in ECR
aws ecr create-repository \
    --repository-name searxng \
    --region us-east-1

# Output will show repository URI:
# 350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng
```

### Build and Push Docker Image

```bash
# Navigate to searxng directory
cd searxng

# Build image for x86_64 architecture (required for ECS Fargate)
docker buildx build --platform linux/amd64 -t searxng-custom:amd64 .

# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    350996086543.dkr.ecr.us-east-1.amazonaws.com

# Tag image for ECR
docker tag searxng-custom:amd64 \
    350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng:amd64

# Push to ECR
docker push 350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng:amd64
```

## Step 3: AWS Infrastructure Setup

### Get Default VPC Information

```bash
# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)

# Get public subnets
aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[?MapPublicIpOnLaunch==`true`].[SubnetId,AvailabilityZone]' \
    --output table
```

### Create Security Group

```bash
# Create security group for SearxNG
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name searxng-public \
    --description "SearxNG public access" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

# Allow HTTP access on port 8080
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0
```

## Step 4: ECS Setup

### Create ECS Cluster

```bash
# Create ECS cluster
aws ecs create-cluster \
    --cluster-name searxng-cluster \
    --region us-east-1
```

### Create CloudWatch Log Group

```bash
# Create log group for container logs
aws logs create-log-group \
    --log-group-name "/ecs/searxng" \
    --region us-east-1
```

### ECS Task Definition

Create `deployment/ecs-task-definition.json`:

```json
{
  "family": "searxng-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::350996086543:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "searxng",
      "image": "350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng:amd64",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp",
          "hostPort": 8080
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/searxng",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {
          "name": "BIND_ADDRESS",
          "value": "0.0.0.0:8080"
        }
      ]
    }
  ]
}
```

### Register Task Definition

```bash
# Register the task definition
aws ecs register-task-definition \
    --cli-input-json file://deployment/ecs-task-definition.json \
    --region us-east-1
```

## Step 5: Deploy ECS Service

### Create ECS Service

```bash
# Replace with your actual subnet IDs and security group ID
aws ecs create-service \
    --cluster searxng-cluster \
    --service-name searxng-service \
    --task-definition searxng-task:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-96b74dcd,subnet-40fa9a25],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
    --region us-east-1
```

### Get Public IP Address

```bash
# Get task ARN
TASK_ARN=$(aws ecs list-tasks \
    --cluster searxng-cluster \
    --service-name searxng-service \
    --query 'taskArns[0]' \
    --output text)

# Get network interface ID
ENI_ID=$(aws ecs describe-tasks \
    --cluster searxng-cluster \
    --tasks $TASK_ARN \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

# Get public IP
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids $ENI_ID \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

echo "SearxNG is accessible at: http://$PUBLIC_IP:8080"
```

## Step 6: Verification

### Test Deployment

```bash
# Test HTTP response
curl -I http://$PUBLIC_IP:8080

# Expected output:
# HTTP/1.1 200 OK
# content-type: text/html; charset=utf-8
# server: granian
```

### View Logs

```bash
# View container logs
aws logs describe-log-streams \
    --log-group-name "/ecs/searxng" \
    --query 'logStreams[0].logStreamName' \
    --output text

# Get recent log events
aws logs get-log-events \
    --log-group-name "/ecs/searxng" \
    --log-stream-name "ecs/searxng/[TASK-ID]" \
    --query 'events[*].message' \
    --output text
```

## Automation Scripts

### Deployment Script

Create `scripts/deploy-searxng-ecs.sh`:

```bash
#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng"
CLUSTER_NAME="searxng-cluster"
SERVICE_NAME="searxng-service"

echo "🚀 Deploying SearxNG to AWS ECS..."

# Build and push image
echo "Building Docker image..."
docker buildx build --platform linux/amd64 -t searxng-custom:amd64 .

echo "Pushing to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin ${ECR_REPOSITORY%/*}

docker tag searxng-custom:amd64 $ECR_REPOSITORY:amd64
docker push $ECR_REPOSITORY:amd64

# Deploy to ECS
echo "Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force-new-deployment \
    --region $AWS_REGION

echo "✅ Deployment complete!"
```

### Cleanup Script

Create `scripts/cleanup-searxng-ecs.sh`:

```bash
#!/bin/bash
set -e

AWS_REGION="us-east-1"
CLUSTER_NAME="searxng-cluster"
SERVICE_NAME="searxng-service"

echo "🧹 Cleaning up SearxNG deployment..."

# Delete ECS service
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --desired-count 0 \
    --region $AWS_REGION

aws ecs delete-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --region $AWS_REGION

# Delete cluster
aws ecs delete-cluster \
    --cluster $CLUSTER_NAME \
    --region $AWS_REGION

# Delete log group
aws logs delete-log-group \
    --log-group-name "/ecs/searxng" \
    --region $AWS_REGION

echo "✅ Cleanup complete!"
```

## Cost Estimation

### ECS Fargate Pricing (us-east-1)

- **CPU**: 512 CPU units = 0.5 vCPU = $0.02048/hour
- **Memory**: 1024 MB = 1 GB = $0.00224/hour
- **Total**: ~$0.023/hour or ~$16.56/month

### Additional Costs

- **ECR Storage**: $0.10/GB/month (minimal for single image)
- **CloudWatch Logs**: $0.50/GB ingested (depends on log volume)
- **Data Transfer**: $0.09/GB out (depends on usage)

## Security Considerations

### Network Security

- Security group restricts access to port 8080 only
- No SSH access configured
- Container runs with non-root user

### Application Security

- Bot detection limiter disabled (no Redis dependency)
- Security headers configured
- No external authentication (consider adding for production)

### Recommendations for Production

1. **Use Application Load Balancer** for SSL termination
2. **Add CloudFront** for caching and DDoS protection
3. **Implement WAF** for additional security
4. **Use private subnets** with NAT gateway
5. **Add auto-scaling** based on CPU/memory usage

## Troubleshooting

### Common Issues

#### Architecture Mismatch
```
Error: exec format error
```
**Solution**: Build image with `--platform linux/amd64` flag

#### Network Connectivity
```
Error: Cannot reach public IP
```
**Solution**: Check security group allows inbound on port 8080

#### Container Startup Failure
```
Error: Valkey database connection
```
**Solution**: Ensure `limiter: false` in settings.yml

#### ECR Authentication
```
Error: no basic auth credentials
```
**Solution**: Re-run ECR login command

### Debugging Commands

```bash
# Check service status
aws ecs describe-services \
    --cluster searxng-cluster \
    --services searxng-service

# Check task status
aws ecs describe-tasks \
    --cluster searxng-cluster \
    --tasks [TASK-ARN]

# View recent logs
aws logs tail /ecs/searxng --follow
```

## File Structure

```
accelerated-discovery/
├── docs/
│   └── searxng-aws-deployment.md
├── deployment/
│   ├── ecs-task-definition.json
│   ├── docker-compose-ecr.yml
│   └── nginx.conf
├── scripts/
│   ├── deploy-searxng-ecs.sh
│   ├── cleanup-searxng-ecs.sh
│   └── push_searxng_to_ecr.sh
└── searxng/
    ├── Dockerfile
    ├── docker-compose.yml
    └── searxng_settings.yml
```

## Next Steps

1. **Domain Setup**: Configure Route 53 domain and SSL certificate
2. **Load Balancer**: Add ALB for production-grade access
3. **Monitoring**: Set up CloudWatch alarms and dashboards
4. **Auto-scaling**: Configure ECS service auto-scaling
5. **Backup**: Implement configuration backup strategy

## References

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [SearxNG Documentation](https://docs.searxng.org/)
- [Docker Multi-platform Builds](https://docs.docker.com/build/building/multi-platform/)