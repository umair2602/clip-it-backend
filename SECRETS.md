# Secret Management Guide

This guide explains how to securely manage secrets for the Clip-It backend application.

## Overview

The application uses a **hybrid approach** for secret management:

- **Local Development**: Uses `.env` file (via `python-dotenv`)
- **Production (AWS)**: Uses AWS Systems Manager (SSM) Parameter Store
- **Automatic Fallback**: The application automatically tries AWS SSM first, then falls back to environment variables

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Code                      │
│                  (config.py)                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Secrets Manager      │
         │  (utils/secrets_      │
         │   manager.py)         │
         └───────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  ┌─────────┐      ┌──────────┐
  │ AWS SSM │      │ .env     │
  │ (Prod)  │      │ (Local)   │
  └─────────┘      └──────────┘
```

## Setup Instructions

### 1. Local Development Setup

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Fill in your secrets in `.env`:
   ```bash
   # Edit .env with your actual values
   nano .env
   ```

3. The application will automatically load from `.env` when running locally.

### 2. Production Setup (AWS)

#### Step 1: Prepare Your Secrets

Create a `.env` file with all your production secrets (or export them as environment variables).

#### Step 2: Store Secrets in AWS SSM

Run the setup script to store all secrets in AWS SSM Parameter Store:

```bash
# Make sure AWS CLI is configured
aws configure

# Run the setup script
./scripts/setup-secrets.sh .env
```

This script will:
- Read all secrets from your `.env` file
- Store them in AWS SSM Parameter Store as `SecureString` parameters
- Use the prefix `/clip-it/` for all parameters

#### Step 3: Verify Secrets

List all stored parameters:
```bash
aws ssm describe-parameters \
  --parameter-filters "Key=Name,Values=/clip-it/" \
  --query 'Parameters[*].[Name,Type]' \
  --output table
```

View a specific secret (decrypted):
```bash
aws ssm get-parameter \
  --name /clip-it/openai-api-key \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text
```

#### Step 4: Update IAM Permissions

Ensure your ECS task execution role has permissions to read SSM parameters:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters",
        "ssm:GetParametersByPath"
      ],
      "Resource": "arn:aws:ssm:*:*:parameter/clip-it/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.*.amazonaws.com"
        }
      }
    }
  ]
}
```

## Secret Parameters

The following secrets are stored in SSM Parameter Store:

| Parameter Name | Environment Variable | Description |
|---------------|---------------------|-------------|
| `/clip-it/openai-api-key` | `OPENAI_API_KEY` | OpenAI API key for content analysis |
| `/clip-it/sieve-api-key` | `SIEVE_API_KEY` | Sieve API key for video processing |
| `/clip-it/hf-token` | `HF_TOKEN` | Hugging Face token |
| `/clip-it/huggingface-token` | `HUGGINGFACE_TOKEN` | Hugging Face token (alternative) |
| `/clip-it/aws-access-key-id` | `AWS_ACCESS_KEY_ID` | AWS access key (use IAM roles in production) |
| `/clip-it/aws-secret-access-key` | `AWS_SECRET_ACCESS_KEY` | AWS secret key (use IAM roles in production) |
| `/clip-it/aws-region` | `AWS_REGION` | AWS region |
| `/clip-it/s3-bucket` | `S3_BUCKET` | S3 bucket name |
| `/clip-it/mongodb-url` | `MONGODB_URL` | MongoDB connection string |
| `/clip-it/mongodb-db-name` | `MONGODB_DB_NAME` | MongoDB database name |
| `/clip-it/jwt-secret-key` | `JWT_SECRET_KEY` | JWT signing secret |
| `/clip-it/tiktok-client-key` | `TIKTOK_CLIENT_KEY` | TikTok OAuth client key |
| `/clip-it/tiktok-client-secret` | `TIKTOK_CLIENT_SECRET` | TikTok OAuth client secret |
| `/clip-it/tiktok-verification-key` | `TIKTOK_VERIFICATION_KEY` | TikTok verification key |
| `/clip-it/tiktok-redirect-uri` | `TIKTOK_REDIRECT_URI` | TikTok OAuth redirect URI |
| `/clip-it/tiktok-scopes` | `TIKTOK_SCOPES` | TikTok OAuth scopes |
| `/clip-it/tiktok-api-base` | `TIKTOK_API_BASE` | TikTok API base URL |
| `/clip-it/tiktok-auth-base` | `TIKTOK_AUTH_BASE` | TikTok Auth base URL |
| `/clip-it/proxy-base-url` | `PROXY_BASE_URL` | Proxy base URL |

## Updating Secrets

### Update a Single Secret

```bash
aws ssm put-parameter \
  --name /clip-it/openai-api-key \
  --value "new-value-here" \
  --type SecureString \
  --overwrite
```

### Update All Secrets from .env

```bash
./scripts/setup-secrets.sh .env
```

The script will overwrite existing parameters with new values.

## Security Best Practices

1. **Never commit `.env` files** - They are in `.gitignore`
2. **Use IAM roles** - In production, use IAM roles instead of access keys
3. **Rotate secrets regularly** - Update secrets periodically
4. **Use separate secrets per environment** - Different values for dev/staging/prod
5. **Enable CloudTrail** - Monitor access to SSM parameters
6. **Use KMS encryption** - SSM SecureString parameters are encrypted by default

## Troubleshooting

### Application can't find secrets

1. **Check AWS credentials**:
   ```bash
   aws sts get-caller-identity
   ```

2. **Verify parameter exists**:
   ```bash
   aws ssm get-parameter --name /clip-it/openai-api-key --with-decryption
   ```

3. **Check IAM permissions** - Ensure the execution role has `ssm:GetParameter` permission

4. **Check region** - Ensure the application is looking in the correct AWS region

### Local development not working

1. Ensure `.env` file exists in the project root
2. Check that `python-dotenv` is installed: `pip install python-dotenv`
3. Verify environment variables are loaded: `python -c "from config import settings; print(settings.OPENAI_API_KEY)"`

## Migration from Hardcoded Secrets

If you have existing hardcoded secrets in your code:

1. Extract all secrets to a `.env` file
2. Run `./scripts/setup-secrets.sh .env` to store in AWS
3. Update your code to use `config.py` settings (already done)
4. Remove hardcoded secrets from code
5. Deploy updated infrastructure

## Additional Resources

- [AWS SSM Parameter Store Documentation](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html)
- [AWS Secrets Manager vs SSM Parameter Store](https://docs.aws.amazon.com/secretsmanager/latest/userguide/integrating_csi_driver.html)
- [ECS Task Execution Role](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_execution_IAM_role.html)

