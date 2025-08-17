#!/bin/bash

# Script to push SearxNG Docker image to AWS ECR
# Usage: ./push_searxng_to_ecr.sh [tag]

set -e

# Configuration
ECR_REPOSITORY="350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng"
AWS_REGION="us-east-1"
IMAGE_TAG="${1:-amd64}"

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