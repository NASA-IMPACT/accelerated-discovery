#!/bin/bash

# Script to push SearxNG Docker image to AWS ECR
# Usage: ./push_searxng_to_ecr.sh [tag] [aws_region] [aws_account_id]

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

# Configuration
IMAGE_TAG="${1:-latest}"
ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/searxng"

echo "🔧 Building SearxNG Docker image for x86_64 architecture..."
cd "$(dirname "$0")/../searxng"

# Build for x86_64 to ensure ECS Fargate compatibility
docker buildx build --platform linux/amd64 -t searxng-custom:${IMAGE_TAG} .

echo "🔐 Authenticating with AWS ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY%/*}

echo "🏷️  Tagging image for ECR..."
docker tag searxng-custom:${IMAGE_TAG} ${ECR_REPOSITORY}:${IMAGE_TAG}

echo "📤 Pushing image to ECR..."
docker push ${ECR_REPOSITORY}:${IMAGE_TAG}

echo "✅ Successfully pushed SearxNG image to ECR!"
echo "📍 Image URI: ${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "🚀 Deployment options:"
echo "   Local: docker run -p 8080:8080 ${ECR_REPOSITORY}:${IMAGE_TAG}"
echo "   ECS: See docs/searxng-aws-deployment.md for full deployment guide"
echo "   Compose: docker-compose -f deployment/docker-compose-ecr.yml up"