#!/bin/bash

# Infrastructure Setup Script
# This script sets up the AWS infrastructure for the first time

set -e

echo "🏗️ Setting up Clip-It AWS Infrastructure..."

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first:"
    echo "   https://aws.amazon.com/cli/"
    exit 1
fi

# Check Node.js for CDK
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install it first:"
    echo "   https://nodejs.org/"
    exit 1
fi

# Check if CDK is installed
if ! command -v cdk &> /dev/null; then
    echo "📦 Installing AWS CDK..."
    npm install -g aws-cdk
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it first."
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install it first."
    exit 1
fi

echo "✅ All prerequisites are installed"

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
pip3 install -r requirements.txt

# Bootstrap CDK
echo "🔧 Bootstrapping CDK (this may take a few minutes)..."
cdk bootstrap

echo "✅ CDK bootstrapped successfully"

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
echo "🎉 Infrastructure Setup Complete!"
echo "================================"
echo "🌐 Application URL: http://$LOAD_BALANCER_URL"
echo "🪣 S3 Bucket: $S3_BUCKET"
echo "🔴 Redis Endpoint: $REDIS_ENDPOINT"
echo ""
echo "📝 Next Steps:"
echo "1. Set up your GitHub repository secrets:"
echo "   - Go to your GitHub repo → Settings → Secrets and variables → Actions"
echo "   - Add the following secrets:"
echo ""
echo "   AWS_ACCESS_KEY_ID = $(aws configure get aws_access_key_id)"
echo "   AWS_SECRET_ACCESS_KEY = [Your AWS Secret Key]"
echo "   MONGODB_URL = [Your MongoDB connection string]"
echo "   OPENAI_API_KEY = [Your OpenAI API key]"
echo "   SIEVE_API_KEY = [Your Sieve API key]"
echo "   TIKTOK_CLIENT_KEY = [Your TikTok client key]"
echo "   TIKTOK_CLIENT_SECRET = [Your TikTok client secret]"
echo "   TIKTOK_REDIRECT_URI = http://$LOAD_BALANCER_URL/tiktok/callback/"
echo "   JWT_SECRET_KEY = [Generate a secure random string]"
echo ""
echo "2. Push your code to trigger the GitHub Actions deployment:"
echo "   git add ."
echo "   git commit -m 'Deploy to AWS'"
echo "   git push origin main"
echo ""
echo "3. Your application will be available at: http://$LOAD_BALANCER_URL"
echo ""
echo "✅ Setup completed successfully!"
