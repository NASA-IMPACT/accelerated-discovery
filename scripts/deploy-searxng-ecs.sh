#!/bin/bash

# Deploy SearxNG to AWS ECS Fargate
# Usage: ./deploy-searxng-ecs.sh [AWS_REGION] [AWS_ACCOUNT_ID]

set -e

# Load configuration from file
CONFIG_FILE="$(dirname "$0")/../deployment/aws-config.env"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Please copy deployment/aws-config.env.example to deployment/aws-config.env and configure it"
    exit 1
fi

source "$CONFIG_FILE"

# Validate required configuration
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "123456789012" ]; then
    echo "❌ AWS_ACCOUNT_ID not configured. Please set it in $CONFIG_FILE"
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
    echo "❌ AWS_REGION not configured. Please set it in $CONFIG_FILE"
    exit 1
fi

# Set defaults for optional values
CLUSTER_NAME="${CLUSTER_NAME:-searxng-cluster}"
SERVICE_NAME="${SERVICE_NAME:-searxng-service}"
TASK_DEFINITION="${TASK_DEFINITION:-searxng-task}"
ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/searxng"

echo "🚀 Deploying SearxNG to AWS ECS Fargate..."

# Check if cluster exists
if ! aws ecs describe-clusters --clusters $CLUSTER_NAME --region $AWS_REGION >/dev/null 2>&1; then
    echo "Creating ECS cluster..."
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $AWS_REGION
fi

# Check if log group exists
if ! aws logs describe-log-groups --log-group-name-prefix "/ecs/searxng" --region $AWS_REGION | grep -q "/ecs/searxng"; then
    echo "Creating CloudWatch log group..."
    aws logs create-log-group --log-group-name "/ecs/searxng" --region $AWS_REGION
fi

# Generate task definition from template
echo "Generating ECS task definition..."
cd "$(dirname "$0")/../deployment"

# Create task definition with current AWS account/region
envsubst < ecs-task-definition-template.json > ecs-task-definition.json

# Register task definition
echo "Registering ECS task definition..."
aws ecs register-task-definition \
    --cli-input-json file://ecs-task-definition.json \
    --region $AWS_REGION

# Check if service exists
if aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION >/dev/null 2>&1; then
    echo "Updating existing ECS service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --task-definition $TASK_DEFINITION \
        --force-new-deployment \
        --region $AWS_REGION
else
    echo "Creating new ECS service..."
    
    # Get default VPC and subnets
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[?MapPublicIpOnLaunch==`true`].SubnetId' --output text --region $AWS_REGION | tr '\t' ',')
    
    # Create security group if it doesn't exist
    SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=searxng-public" --query 'SecurityGroups[0].GroupId' --output text --region $AWS_REGION 2>/dev/null || echo "None")
    
    if [ "$SG_ID" = "None" ]; then
        echo "Creating security group..."
        SG_ID=$(aws ec2 create-security-group \
            --group-name searxng-public \
            --description "SearxNG public access" \
            --vpc-id $VPC_ID \
            --query 'GroupId' \
            --output text \
            --region $AWS_REGION)
        
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port 8080 \
            --cidr 0.0.0.0/0 \
            --region $AWS_REGION
    fi
    
    # Create service
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_DEFINITION \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --region $AWS_REGION
fi

echo "⏳ Waiting for service to stabilize..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $AWS_REGION

# Get public IP
echo "🔍 Getting public IP address..."
TASK_ARN=$(aws ecs list-tasks \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --query 'taskArns[0]' \
    --output text \
    --region $AWS_REGION)

if [ "$TASK_ARN" != "None" ]; then
    ENI_ID=$(aws ecs describe-tasks \
        --cluster $CLUSTER_NAME \
        --tasks $TASK_ARN \
        --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
        --output text \
        --region $AWS_REGION)
    
    PUBLIC_IP=$(aws ec2 describe-network-interfaces \
        --network-interface-ids $ENI_ID \
        --query 'NetworkInterfaces[0].Association.PublicIp' \
        --output text \
        --region $AWS_REGION)
    
    echo "✅ SearxNG deployment complete!"
    echo "🌐 Public URL: http://$PUBLIC_IP:8080"
    echo "📊 Logs: aws logs tail /ecs/searxng --follow --region $AWS_REGION"
else
    echo "⚠️  Task not running yet. Check ECS console for status."
fi

echo "🎯 Deployment Summary:"
echo "   Cluster: $CLUSTER_NAME"
echo "   Service: $SERVICE_NAME"
echo "   Task Definition: $TASK_DEFINITION"
echo "   Region: $AWS_REGION"