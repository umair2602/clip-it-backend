# Clip-It AWS Setup Guide

This guide will help you set up the Clip-It application on AWS from scratch.

## 🎯 What You'll Get

After following this guide, you'll have:
- ✅ A fully deployed Clip-It application on AWS
- ✅ Automatic deployments via GitHub Actions
- ✅ Scalable infrastructure with auto-scaling
- ✅ Monitoring and logging setup
- ✅ Cost-optimized configuration

## 📋 Prerequisites

### Required Tools
1. **AWS Account** - [Create one here](https://aws.amazon.com/)
2. **AWS CLI** - [Install here](https://aws.amazon.com/cli/)
3. **Node.js** - [Install here](https://nodejs.org/)
4. **Python 3.7+** - [Install here](https://python.org/)
5. **Git** - [Install here](https://git-scm.com/)
6. **Docker** - [Install here](https://docker.com/)

### Required Services
- **MongoDB Atlas** - [Sign up here](https://www.mongodb.com/atlas)
- **OpenAI API** - [Get API key here](https://platform.openai.com/)
- **Sieve API** - [Sign up here](https://sieve.ai/)
- **TikTok Developer** - [Apply here](https://developers.tiktok.com/)

## 🚀 Step-by-Step Setup

### Step 1: AWS Account Setup

1. **Create AWS Account**
   - Go to [aws.amazon.com](https://aws.amazon.com/)
   - Click "Create an AWS Account"
   - Follow the signup process

2. **Create IAM User**
   ```bash
   # Login to AWS Console
   # Go to IAM → Users → Create User
   # User name: clip-it-deployer
   # Attach policies: AdministratorAccess (for simplicity)
   # Create access key and download credentials
   ```

3. **Configure AWS CLI**
   ```bash
   aws configure
   # Enter your Access Key ID
   # Enter your Secret Access Key
   # Enter region: us-east-1
   # Enter output format: json
   ```

### Step 2: Install Required Tools

#### Install AWS CDK
```bash
npm install -g aws-cdk
```

#### Verify Installation
```bash
aws --version
cdk --version
python3 --version
node --version
```

### Step 3: Clone and Setup Repository

```bash
# Clone your repository
git clone <your-repo-url>
cd clip-it-backend

# Make scripts executable
chmod +x scripts/*.sh
```

### Step 4: Deploy Infrastructure

```bash
# Run the setup script
./scripts/setup-infrastructure.sh
```

This will:
- ✅ Check all prerequisites
- ✅ Install Python dependencies
- ✅ Bootstrap CDK
- ✅ Deploy all AWS resources
- ✅ Show you the outputs

### Step 5: Configure External Services

#### MongoDB Atlas Setup
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a new cluster
3. Create a database user
4. Get connection string
5. Whitelist your IP (or use 0.0.0.0/0 for AWS)

#### OpenAI API Setup
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create API key
3. Note down the key

#### Sieve API Setup
1. Go to [Sieve](https://sieve.ai/)
2. Sign up and get API key
3. Note down the key

#### TikTok Developer Setup
1. Go to [TikTok Developers](https://developers.tiktok.com/)
2. Create an app
3. Get Client Key and Client Secret
4. Set redirect URI to: `http://your-load-balancer-url/tiktok/callback/`

### Step 6: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Add the following secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | Your AWS Access Key |
| `AWS_SECRET_ACCESS_KEY` | `...` | Your AWS Secret Key |
| `MONGODB_URL` | `mongodb+srv://...` | MongoDB connection string |
| `OPENAI_API_KEY` | `sk-...` | OpenAI API key |
| `SIEVE_API_KEY` | `...` | Sieve API key |
| `TIKTOK_CLIENT_KEY` | `...` | TikTok client key |
| `TIKTOK_CLIENT_SECRET` | `...` | TikTok client secret |
| `TIKTOK_REDIRECT_URI` | `http://your-lb-url/tiktok/callback/` | TikTok redirect URI |
| `JWT_SECRET_KEY` | `your-secure-random-string` | JWT signing secret |

### Step 7: Deploy Application

```bash
# Commit and push to trigger deployment
git add .
git commit -m "Deploy to AWS"
git push origin main
```

### Step 8: Verify Deployment

1. **Check GitHub Actions**
   - Go to your repo → Actions tab
   - Watch the deployment progress

2. **Check AWS Console**
   - ECS → Clusters → clip-it-cluster
   - Check services are running

3. **Test Application**
   - Visit your load balancer URL
   - Check `/health` endpoint

## 🔧 Configuration Details

### Environment Variables

The application uses these environment variables:

```bash
# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# Database
MONGODB_URL=mongodb+srv://...

# APIs
OPENAI_API_KEY=sk-...
SIEVE_API_KEY=...

# TikTok OAuth
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...
TIKTOK_REDIRECT_URI=http://...

# Security
JWT_SECRET_KEY=...
```

### Infrastructure Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CloudFront    │────│  Application     │────│   ECS Fargate   │
│   (CDN)         │    │  Load Balancer   │    │   (Web App)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                       ┌──────────────────┐            │
                       │  ElastiCache     │◄───────────┘
                       │  (Redis)         │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │   ECS Fargate    │
                       │   (Workers)      │
                       └──────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│      S3         │◄───│   Workers       │───►│   MongoDB      │
│   (Storage)     │    │   (Processing)  │    │   Atlas        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Monitoring

### CloudWatch Logs
- **Log Group**: `/ecs/clip-it`
- **Retention**: 7 days
- **Streams**: Separate for web and worker

### Health Checks
- **Web Service**: `GET /health`
- **Worker Service**: Redis connection check
- **Load Balancer**: HTTP health checks

### Metrics
- CPU utilization
- Memory usage
- Request count
- Error rates

## 💰 Cost Management

### Estimated Monthly Costs

| Service | Low Usage | Medium Usage | High Usage |
|---------|-----------|--------------|------------|
| ECS Fargate (Web) | $30 | $80 | $150 |
| ECS Fargate (Workers) | $200 | $500 | $800 |
| ElastiCache Redis | $15 | $30 | $50 |
| Application Load Balancer | $20 | $30 | $50 |
| S3 Storage | $10 | $50 | $100 |
| CloudWatch | $20 | $40 | $60 |
| **Total** | **$295** | **$730** | **$1,210** |

### Cost Optimization Tips
1. **Use Spot Instances** for workers (60-90% savings)
2. **S3 Intelligent Tiering** for storage optimization
3. **Reserved Instances** for predictable workloads
4. **CloudWatch Logs Retention** (reduce from 7 days)

## 🚨 Troubleshooting

### Common Issues

#### 1. CDK Bootstrap Fails
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check CDK version
cdk --version
```

#### 2. Deployment Fails
```bash
# Check ECS service status
aws ecs describe-services --cluster clip-it-cluster --services clip-it-web-service

# Check task logs
aws logs get-log-events --log-group-name /ecs/clip-it --log-stream-name web/...
```

#### 3. Redis Connection Issues
```bash
# Check Redis cluster status
aws elasticache describe-cache-clusters --cache-cluster-id clip-it-redis-cluster
```

#### 4. Load Balancer Health Checks Failing
```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn your-target-group-arn
```

### Debug Commands
```bash
# View ECS service events
aws ecs describe-services --cluster clip-it-cluster --services clip-it-web-service --query 'services[0].events'

# Check task definition
aws ecs describe-task-definition --task-definition clip-it-web-task

# View CloudWatch logs
aws logs describe-log-streams --log-group-name /ecs/clip-it
```

## 🔄 Updates and Maintenance

### Updating the Application
1. Make code changes
2. Commit and push to main branch
3. GitHub Actions automatically deploys

### Updating Infrastructure
```bash
# Modify infrastructure/app.py
# Deploy changes
cd infrastructure
cdk deploy
```

### Scaling
```bash
# Scale web service
aws ecs update-service --cluster clip-it-cluster --service clip-it-web-service --desired-count 3

# Scale worker service
aws ecs update-service --cluster clip-it-cluster --service clip-it-worker-service --desired-count 2
```

## 🗑️ Cleanup

### Destroy Infrastructure
```bash
./scripts/destroy.sh
```

### Manual Cleanup
```bash
# Delete ECS services
aws ecs delete-service --cluster clip-it-cluster --service clip-it-web-service --force
aws ecs delete-service --cluster clip-it-cluster --service clip-it-worker-service --force

# Delete ECR repositories
aws ecr delete-repository --repository-name clip-it-web --force
aws ecr delete-repository --repository-name clip-it-worker --force

# Delete S3 bucket (must be empty)
aws s3 rm s3://your-bucket-name --recursive
aws s3api delete-bucket --bucket your-bucket-name
```

## 📞 Support

### Getting Help
1. Check CloudWatch logs for errors
2. Review ECS service events
3. Check GitHub Actions logs
4. Verify environment variables

### Useful Commands
```bash
# Get all outputs
aws cloudformation describe-stacks --stack-name ClipItStack --query 'Stacks[0].Outputs'

# Check service status
aws ecs list-services --cluster clip-it-cluster

# View recent logs
aws logs tail /ecs/clip-it --follow
```

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

---

**Happy Deploying! 🚀**

If you encounter any issues, check the troubleshooting section or create an issue in the repository.
