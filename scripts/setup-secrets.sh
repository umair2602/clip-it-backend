#!/bin/bash

# Script to securely store secrets in AWS SSM Parameter Store
# This script reads from a .env file and stores secrets in AWS

set -e

echo "🔐 Setting up secrets in AWS SSM Parameter Store..."

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

# Check if .env file exists
ENV_FILE="${1:-.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    echo "Usage: $0 [path-to-env-file]"
    exit 1
fi

echo "📄 Reading secrets from: $ENV_FILE"

# Function to set a parameter in SSM
set_parameter() {
    local param_name=$1
    local param_value=$2
    local description=$3
    
    if [ -z "$param_value" ]; then
        echo "⚠️  Skipping $param_name (empty value)"
        return
    fi
    
    # Check if parameter already exists
    existing_param=$(aws ssm describe-parameters \
        --region "$AWS_REGION" \
        --parameter-filters "Key=Name,Values=$param_name" \
        --query 'Parameters[0].Type' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$existing_param" != "None" ] && [ -n "$existing_param" ]; then
        # Parameter exists - preserve its type
        echo "🔒 Updating existing parameter $param_name (type: $existing_param)..."
        aws ssm put-parameter \
            --region "$AWS_REGION" \
            --name "$param_name" \
            --value "$param_value" \
            --type "$existing_param" \
            --description "$description" \
            --overwrite \
            --no-cli-pager > /dev/null 2>&1 || {
            echo "⚠️  Failed to update $param_name"
            return 1
        }
    else
        # Parameter doesn't exist - create as SecureString
        echo "🔒 Creating new parameter $param_name as SecureString..."
        aws ssm put-parameter \
            --region "$AWS_REGION" \
            --name "$param_name" \
            --value "$param_value" \
            --type "SecureString" \
            --description "$description" \
            --no-cli-pager > /dev/null 2>&1 || {
            echo "⚠️  Failed to create $param_name"
            return 1
        }
    fi
    echo "✅ Set $param_name"
}

# Read .env file and set parameters
while IFS='=' read -r key value || [ -n "$key" ]; do
    # Skip empty lines and comments
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    
    # Remove leading/trailing whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    # Remove quotes if present
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    
    # Skip if value is empty
    [[ -z "$value" ]] && continue
    
    # Map environment variable names to SSM parameter names
    case "$key" in
        OPENAI_API_KEY)
            set_parameter "/clip-it/openai-api-key" "$value" "OpenAI API Key for content analysis"
            ;;
        SIEVE_API_KEY)
            set_parameter "/clip-it/sieve-api-key" "$value" "Sieve API Key for video processing"
            ;;
        HF_TOKEN)
            set_parameter "/clip-it/hf-token" "$value" "Hugging Face token"
            ;;
        HUGGINGFACE_TOKEN)
            set_parameter "/clip-it/huggingface-token" "$value" "Hugging Face token (alternative)"
            ;;
        AWS_ACCESS_KEY_ID)
            set_parameter "/clip-it/aws-access-key-id" "$value" "AWS Access Key ID"
            ;;
        AWS_SECRET_ACCESS_KEY)
            set_parameter "/clip-it/aws-secret-access-key" "$value" "AWS Secret Access Key"
            ;;
        AWS_REGION)
            set_parameter "/clip-it/aws-region" "$value" "AWS Region"
            ;;
        S3_BUCKET)
            set_parameter "/clip-it/s3-bucket" "$value" "S3 Bucket name"
            ;;
        MONGODB_URL)
            set_parameter "/clip-it/mongodb-url" "$value" "MongoDB connection URL"
            ;;
        MONGODB_DB_NAME)
            set_parameter "/clip-it/mongodb-db-name" "$value" "MongoDB database name"
            ;;
        JWT_SECRET_KEY)
            set_parameter "/clip-it/jwt-secret-key" "$value" "JWT secret key for authentication"
            ;;
        TIKTOK_CLIENT_KEY)
            set_parameter "/clip-it/tiktok-client-key" "$value" "TikTok OAuth client key"
            ;;
        TIKTOK_CLIENT_SECRET)
            set_parameter "/clip-it/tiktok-client-secret" "$value" "TikTok OAuth client secret"
            ;;
        TIKTOK_VERIFICATION_KEY)
            set_parameter "/clip-it/tiktok-verification-key" "$value" "TikTok verification key"
            ;;
        TIKTOK_REDIRECT_URI)
            set_parameter "/clip-it/tiktok-redirect-uri" "$value" "TikTok OAuth redirect URI"
            ;;
        TIKTOK_SCOPES)
            set_parameter "/clip-it/tiktok-scopes" "$value" "TikTok OAuth scopes"
            ;;
        TIKTOK_API_BASE)
            set_parameter "/clip-it/tiktok-api-base" "$value" "TikTok API base URL"
            ;;
        TIKTOK_AUTH_BASE)
            set_parameter "/clip-it/tiktok-auth-base" "$value" "TikTok Auth base URL"
            ;;
        PROXY_BASE_URL)
            set_parameter "/clip-it/proxy-base-url" "$value" "Proxy base URL"
            ;;
        *)
            echo "⚠️  Unknown environment variable: $key (skipping)"
            ;;
    esac
done < "$ENV_FILE"

echo ""
echo "✅ Secrets setup completed!"
echo ""
echo "📋 To verify, list parameters:"
echo "   aws ssm describe-parameters --region $AWS_REGION --parameter-filters 'Key=Name,Values=/clip-it/'"
echo ""
echo "🔍 To view a specific parameter (decrypted):"
echo "   aws ssm get-parameter --region $AWS_REGION --name /clip-it/openai-api-key --with-decryption --query 'Parameter.Value' --output text"
echo ""
echo "⚠️  Note: Make sure your ECS task execution role has permissions to read these parameters:"
echo "   - ssm:GetParameter"
echo "   - ssm:GetParameters"
echo "   - ssm:GetParametersByPath"
echo "   - kms:Decrypt (if using KMS encryption)"

