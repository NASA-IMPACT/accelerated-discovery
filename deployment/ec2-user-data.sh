#!/bin/bash

# EC2 User Data Script to deploy SearxNG
yum update -y
yum install -y docker

# Start Docker service
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 350996086543.dkr.ecr.us-east-1.amazonaws.com

# Pull and run SearxNG
docker pull 350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng:fixed
docker run -d --name searxng --restart unless-stopped -p 80:8080 350996086543.dkr.ecr.us-east-1.amazonaws.com/searxng:fixed

# Install nginx for SSL termination (optional)
amazon-linux-extras install nginx1 -y
systemctl start nginx
systemctl enable nginx

echo "SearxNG deployed on EC2!"