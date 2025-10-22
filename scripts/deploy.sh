#!/bin/bash

# AWS Deployment Script for Clip-It Application
# This script sets up the AWS infrastructure and deploys the application

set -e

echo "🚀 Starting Clip-It AWS Deployment..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if CDK is installed
if ! command -v cdk &> /dev/null; then
    echo "❌ AWS CDK is not installed. Installing..."
    npm install -g aws-cdk
fi

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials configured"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRASTRUCTURE_DIR="$PROJECT_ROOT/infrastructure"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Infrastructure dir: $INFRASTRUCTURE_DIR"

# Navigate to infrastructure directory
cd "$INFRASTRUCTURE_DIR"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Bootstrap CDK (if not already bootstrapped)
echo "🔧 Bootstrapping CDK..."
cdk bootstrap

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."
cdk deploy --require-approval never

echo "✅ Infrastructure deployment completed!"

# Get outputs
echo "📋 Getting deployment outputs..."
LOAD_BALANCER_URL=$(aws ssm get-parameter --name "/clip-it/load-balancer-dns" --query "Parameter.Value" --output text)
S3_BUCKET=$(aws ssm get-parameter --name "/clip-it/s3-bucket-name" --query "Parameter.Value" --output text)
REDIS_ENDPOINT=$(aws ssm get-parameter --name "/clip-it/redis-endpoint" --query "Parameter.Value" --output text)

echo ""
echo "🎉 Deployment Summary:"
echo "======================"
echo "🌐 Application URL: http://$LOAD_BALANCER_URL"
echo "🪣 S3 Bucket: $S3_BUCKET"
echo "🔴 Redis Endpoint: $REDIS_ENDPOINT"
echo ""
echo "📝 Next Steps:"
echo "1. Set up your environment variables in GitHub Secrets:"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY"
echo "   - MONGODB_URL"
echo "   - OPENAI_API_KEY"
echo "   - SIEVE_API_KEY"
echo "   - TIKTOK_CLIENT_KEY"
echo "   - TIKTOK_CLIENT_SECRET"
echo "   - TIKTOK_REDIRECT_URI"
echo "   - JWT_SECRET_KEY"
echo ""
echo "2. Push your code to trigger the GitHub Actions deployment"
echo ""
echo "3. Your application will be available at: http://$LOAD_BALANCER_URL"
echo ""
echo "✅ Deployment completed successfully!"
