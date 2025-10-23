#!/bin/bash

# Manual Infrastructure Deployment Script
# This script deploys infrastructure using temporary AWS credentials

set -e

echo "🏗️ Deploying Clip-It Infrastructure..."

# Check if required tools are installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

if ! command -v cdk &> /dev/null; then
    echo "📦 Installing AWS CDK..."
    npm install -g aws-cdk
fi

# Set temporary AWS credentials (these will be used for this session only)


echo "🔐 Using temporary AWS credentials for deployment"

# Navigate to infrastructure directory
cd infrastructure

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Bootstrap CDK
echo "🔧 Bootstrapping CDK..."
cdk bootstrap

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."
cdk deploy --require-approval never

echo "✅ Infrastructure deployment completed!"

# Get outputs
echo "📋 Getting deployment outputs..."
LOAD_BALANCER_URL=$(aws ssm get-parameter --name "/clip-it/load-balancer-dns" --query "Parameter.Value" --output text 2>/dev/null || echo "Not available yet")
S3_BUCKET=$(aws ssm get-parameter --name "/clip-it/s3-bucket-name" --query "Parameter.Value" --output text 2>/dev/null || echo "Not available yet")
REDIS_ENDPOINT=$(aws ssm get-parameter --name "/clip-it/redis-endpoint" --query "Parameter.Value" --output text 2>/dev/null || echo "Not available yet")

echo ""
echo "🎉 Infrastructure Deployment Complete!"
echo "======================================"
echo "🌐 Application URL: http://$LOAD_BALANCER_URL"
echo "🪣 S3 Bucket: $S3_BUCKET"
echo "🔴 Redis Endpoint: $REDIS_ENDPOINT"
echo ""
echo "📝 Next Steps:"
echo "1. Add your application secrets to GitHub:"
echo "   - MONGODB_URL"
echo "   - OPENAI_API_KEY"
echo "   - SIEVE_API_KEY"
echo "   - TIKTOK_CLIENT_KEY"
echo "   - TIKTOK_CLIENT_SECRET"
echo "   - TIKTOK_REDIRECT_URI"
echo "   - JWT_SECRET_KEY"
echo ""
echo "2. Push your code to trigger application deployment:"
echo "   git add ."
echo "   git commit -m 'Deploy application'"
echo "   git push origin main"
echo ""
echo "3. Your application will be available at: http://$LOAD_BALANCER_URL"
echo ""
echo "✅ Deployment completed successfully!"
