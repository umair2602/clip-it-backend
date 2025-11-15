#!/bin/bash

# Script to request an SSL certificate from AWS Certificate Manager
# This certificate can be used to enable HTTPS on your Application Load Balancer

set -e

echo "🔐 Requesting SSL Certificate from AWS Certificate Manager..."

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

# Get domain name from user or use default
if [ -z "$1" ]; then
    echo ""
    echo "📝 Please provide a domain name for the certificate."
    echo "   Examples:"
    echo "   - api.klipz.ai (for API subdomain)"
    echo "   - *.klipz.ai (wildcard for all subdomains)"
    echo "   - klipz.ai (for root domain)"
    echo ""
    read -p "Enter domain name: " DOMAIN_NAME
else
    DOMAIN_NAME=$1
fi

if [ -z "$DOMAIN_NAME" ]; then
    echo "❌ Domain name is required"
    exit 1
fi

# Determine if it's a wildcard or specific domain
if [[ "$DOMAIN_NAME" == *"*"* ]]; then
    echo "🌐 Requesting wildcard certificate for: $DOMAIN_NAME"
    SUBJECT_ALTERNATIVE_NAMES=""
else
    # For non-wildcard, we can add both the domain and www subdomain
    ROOT_DOMAIN=$(echo "$DOMAIN_NAME" | sed 's/^[^.]*\.//')  # Extract root domain
    if [[ "$DOMAIN_NAME" != "www.$ROOT_DOMAIN" ]] && [[ "$DOMAIN_NAME" != "$ROOT_DOMAIN" ]]; then
        # It's a subdomain, add root domain as alternative
        SUBJECT_ALTERNATIVE_NAMES="--subject-alternative-names $ROOT_DOMAIN"
        echo "🌐 Requesting certificate for: $DOMAIN_NAME (with $ROOT_DOMAIN as alternative)"
    else
        SUBJECT_ALTERNATIVE_NAMES=""
        echo "🌐 Requesting certificate for: $DOMAIN_NAME"
    fi
fi

# Request the certificate
echo ""
echo "📋 Requesting certificate..."
CERT_ARN=$(aws acm request-certificate \
    --domain-name "$DOMAIN_NAME" \
    $SUBJECT_ALTERNATIVE_NAMES \
    --validation-method DNS \
    --region "$AWS_REGION" \
    --query 'CertificateArn' \
    --output text)

if [ -z "$CERT_ARN" ] || [ "$CERT_ARN" == "None" ]; then
    echo "❌ Failed to request certificate"
    exit 1
fi

echo "✅ Certificate requested successfully!"
echo ""
echo "📋 Certificate ARN: $CERT_ARN"
echo ""

# Get validation records
echo "⏳ Waiting for validation records to be available..."
sleep 3

VALIDATION_RECORDS=$(aws acm describe-certificate \
    --certificate-arn "$CERT_ARN" \
    --region "$AWS_REGION" \
    --query 'Certificate.DomainValidationOptions[*].[DomainName,ResourceRecord.Name,ResourceRecord.Value,ResourceRecord.Type]' \
    --output text)

if [ -z "$VALIDATION_RECORDS" ]; then
    echo "⚠️  Validation records not yet available. Please check AWS Console."
    echo "   Certificate ARN: $CERT_ARN"
    exit 0
fi

echo ""
echo "📝 DNS VALIDATION REQUIRED"
echo "=========================="
echo ""
echo "You need to add the following CNAME records to your DNS provider:"
echo ""
echo "Certificate ARN: $CERT_ARN"
echo ""

# Parse and display validation records
echo "$VALIDATION_RECORDS" | while IFS=$'\t' read -r domain name value type; do
    echo "For domain: $domain"
    echo "  Type: $type"
    echo "  Name: $name"
    echo "  Value: $value"
    echo ""
done

echo "📋 Instructions:"
echo "1. Go to your DNS provider (Route 53, Cloudflare, etc.)"
echo "2. Add the CNAME records shown above"
echo "3. Wait 5-30 minutes for DNS propagation"
echo "4. AWS will automatically validate the certificate"
echo ""
echo "🔍 To check validation status:"
echo "   aws acm describe-certificate --certificate-arn $CERT_ARN --region $AWS_REGION --query 'Certificate.Status' --output text"
echo ""
echo "✅ Once validated, use this ARN to deploy with HTTPS:"
echo "   export SSL_CERTIFICATE_ARN=$CERT_ARN"
echo "   cd infrastructure && cdk deploy"
echo ""
echo "💾 You can also store it in SSM:"
echo "   aws ssm put-parameter \\"
echo "     --name /clip-it/ssl-certificate-arn \\"
echo "     --value \"$CERT_ARN\" \\"
echo "     --type String \\"
echo "     --description \"SSL Certificate ARN for ALB\""
echo ""


