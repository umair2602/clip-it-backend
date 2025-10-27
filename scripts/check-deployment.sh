#!/bin/bash

# AWS Deployment Status Checker
# This script helps you monitor your deployment progress

echo "🔍 Checking Clip-It AWS Deployment Status..."
echo "=============================================="

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS credentials configured"
echo ""

# Check CloudFormation stack status
echo "📋 CloudFormation Stack Status:"
echo "==============================="
aws cloudformation describe-stacks --stack-name ClipItStack --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "Stack not found or still creating..."

echo ""

# Check ECS cluster
echo "🐳 ECS Cluster Status:"
echo "======================"
aws ecs describe-clusters --clusters clip-it-cluster --query 'clusters[0].status' --output text 2>/dev/null || echo "Cluster not found..."

echo ""

# Check ECS services
echo "🚀 ECS Services Status:"
echo "======================"
aws ecs list-services --cluster clip-it-cluster --query 'serviceArns' --output table 2>/dev/null || echo "No services found..."

echo ""

# Check ECR repositories
echo "📦 ECR Repositories:"
echo "===================="
aws ecr describe-repositories --query 'repositories[].repositoryName' --output table 2>/dev/null || echo "No repositories found..."

echo ""

# Check S3 bucket
echo "🪣 S3 Bucket Status:"
echo "==================="
aws s3 ls | grep clip-it || echo "No clip-it bucket found..."

echo ""

# Check ElastiCache
echo "🔴 Redis Cluster Status:"
echo "========================"
aws elasticache describe-cache-clusters --query 'CacheClusters[0].CacheClusterStatus' --output text 2>/dev/null || echo "Redis cluster not found..."

echo ""
echo "📝 Next Steps:"
echo "1. If CloudFormation shows 'CREATE_IN_PROGRESS', wait for it to complete"
echo "2. If ECS services show 'PENDING', they're still starting up"
echo "3. If you see errors, check the GitHub Actions logs"
echo "4. Once complete, you'll get a Load Balancer URL"
