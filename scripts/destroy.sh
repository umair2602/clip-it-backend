#!/bin/bash

# AWS Infrastructure Destruction Script
# This script destroys the AWS infrastructure

set -e

echo "🗑️ Starting Clip-It Infrastructure Destruction..."

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

# Confirm destruction
echo "⚠️ WARNING: This will destroy all AWS resources created for Clip-It!"
echo "This includes:"
echo "- ECS Cluster and Services"
echo "- Application Load Balancer"
echo "- ElastiCache Redis Cluster"
echo "- S3 Bucket (if empty)"
echo "- All associated networking and security groups"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Destruction cancelled."
    exit 1
fi

# Destroy infrastructure
echo "🏗️ Destroying infrastructure..."
cdk destroy --force

echo "✅ Infrastructure destruction completed!"
echo ""
echo "📝 Note: Some resources may take a few minutes to be fully deleted."
echo "Check the AWS Console to verify all resources have been removed."
