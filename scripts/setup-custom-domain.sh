#!/bin/bash

# Script to set up custom domain for the ALB
# This will point api.klipz.ai to your Application Load Balancer

set -e

echo "🌐 Setting up custom domain for ALB..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS region
AWS_REGION=${AWS_REGION:-$(aws configure get region)}
if [ -z "$AWS_REGION" ]; then
    AWS_REGION="us-east-1"
fi

echo "✅ Using AWS region: $AWS_REGION"

# Get ALB DNS name from SSM or CloudFormation
ALB_DNS=$(aws ssm get-parameter \
    --name "/clip-it/load-balancer-dns" \
    --region "$AWS_REGION" \
    --query "Parameter.Value" \
    --output text 2>/dev/null || echo "")

if [ -z "$ALB_DNS" ]; then
    echo "⚠️  Could not find ALB DNS in SSM. Getting from CloudFormation..."
    ALB_DNS=$(aws cloudformation describe-stacks \
        --stack-name ClipItStack \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerURL'].OutputValue" \
        --output text 2>/dev/null | sed 's|https\?://||' || echo "")
fi

if [ -z "$ALB_DNS" ]; then
    echo "❌ Could not find ALB DNS name. Please provide it manually:"
    read -p "Enter ALB DNS name: " ALB_DNS
fi

echo "✅ ALB DNS: $ALB_DNS"
echo ""

# Get ALB hosted zone ID (required for Route 53 alias)
echo "🔍 Getting ALB hosted zone ID..."
ALB_ARN=$(aws elbv2 describe-load-balancers \
    --region "$AWS_REGION" \
    --query "LoadBalancers[?DNSName=='$ALB_DNS'].LoadBalancerArn" \
    --output text)

if [ -z "$ALB_ARN" ]; then
    echo "❌ Could not find ALB. Please check the DNS name."
    exit 1
fi

ALB_HOSTED_ZONE_ID=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns "$ALB_ARN" \
    --region "$AWS_REGION" \
    --query "LoadBalancers[0].CanonicalHostedZoneId" \
    --output text)

echo "✅ ALB Hosted Zone ID: $ALB_HOSTED_ZONE_ID"
echo ""

# Check if using Route 53
echo "📋 DNS Configuration Instructions:"
echo "=================================="
echo ""
echo "To point api.klipz.ai to your ALB, add this DNS record:"
echo ""
echo "If using Route 53:"
echo "  Type: A (Alias)"
echo "  Name: api.klipz.ai"
echo "  Alias: Yes"
echo "  Alias Target: $ALB_DNS"
echo "  Hosted Zone ID: $ALB_HOSTED_ZONE_ID"
echo ""
echo "If using another DNS provider (Cloudflare, etc.):"
echo "  Type: CNAME"
echo "  Name: api"
echo "  Value: $ALB_DNS"
echo "  TTL: 300 (or Auto)"
echo ""
echo "⚠️  Note: After adding the DNS record, wait 5-30 minutes for propagation."
echo ""
echo "✅ Once DNS is configured, you can access your API at:"
echo "   https://api.klipz.ai"
echo ""

# Optionally create Route 53 record if hosted zone exists
read -p "Do you want to automatically create the Route 53 record? (y/n): " CREATE_R53

if [ "$CREATE_R53" = "y" ] || [ "$CREATE_R53" = "Y" ]; then
    # Find hosted zone for klipz.ai
    HOSTED_ZONE_ID=$(aws route53 list-hosted-zones \
        --query "HostedZones[?Name=='klipz.ai.'].Id" \
        --output text 2>/dev/null | sed 's|/hostedzone/||' || echo "")
    
    if [ -z "$HOSTED_ZONE_ID" ]; then
        echo "⚠️  Could not find Route 53 hosted zone for klipz.ai"
        echo "   Please create the DNS record manually using the instructions above."
        exit 0
    fi
    
    echo "✅ Found hosted zone: $HOSTED_ZONE_ID"
    echo "📝 Creating Route 53 record..."
    
    # Create Route 53 record
    CHANGE_BATCH=$(cat <<EOF
{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "api.klipz.ai",
            "Type": "A",
            "AliasTarget": {
                "DNSName": "$ALB_DNS",
                "HostedZoneId": "$ALB_HOSTED_ZONE_ID",
                "EvaluateTargetHealth": false
            }
        }
    }]
}
EOF
)
    
    CHANGE_ID=$(aws route53 change-resource-record-sets \
        --hosted-zone-id "$HOSTED_ZONE_ID" \
        --change-batch "$CHANGE_BATCH" \
        --query "ChangeInfo.Id" \
        --output text)
    
    echo "✅ Route 53 record created! Change ID: $CHANGE_ID"
    echo "⏳ Waiting for DNS propagation (this may take a few minutes)..."
fi

