#!/bin/bash

# Deploy SearxNG to AWS ECS with Application Load Balancer
set -e

AWS_REGION="us-east-1"
CLUSTER_NAME="searxng-cluster"
SERVICE_NAME="searxng-service"
TASK_DEFINITION="searxng-task"

echo "🚀 Deploying SearxNG to AWS ECS..."

# Create ECS cluster
echo "Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --region $AWS_REGION || true

# Create CloudWatch log group
echo "Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name "/ecs/searxng" \
    --region $AWS_REGION || true

# Register task definition
echo "Registering ECS task definition..."
aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition.json \
    --region $AWS_REGION

# Create ECS service with load balancer
echo "Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_DEFINITION \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
    --region $AWS_REGION

echo "✅ SearxNG deployed to ECS!"
echo "📍 Check the ECS console for the public IP or configure an ALB"