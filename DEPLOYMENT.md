# Clip-It AWS Deployment Guide

This guide will help you deploy the Clip-It application to AWS using ECS Fargate, Redis, and S3.

## 🏗️ Architecture Overview

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
                       │   + GPU Support  │
                       └──────────────────┘
                                │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│      S3         │◄───│   Workers       │───►│   MongoDB      │
│   (Storage)     │    │   (Processing)  │    │   Atlas        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### Required Tools
- [AWS CLI](https://aws.amazon.com/cli/) (v2.0+)
- [AWS CDK](https://aws.amazon.com/cdk/) (v2.0+)
- [Docker](https://www.docker.com/) (for local testing)
- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/) (for CDK)

### AWS Account Setup
1. Create an AWS account
2. Set up IAM user with appropriate permissions
3. Configure AWS CLI: `aws configure`

### Required Permissions
Your AWS user needs the following permissions:
- ECS (Elastic Container Service)
- ECR (Elastic Container Registry)
- ElastiCache
- S3
- VPC
- IAM
- CloudFormation
- Application Load Balancer
- Auto Scaling

## 🚀 Quick Start Deployment

### Step 1: Clone and Setup
```bash
git clone <your-repo-url>
cd clip-it-backend
```

### Step 2: Deploy Infrastructure
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy infrastructure
./scripts/deploy.sh
```

### Step 3: Configure GitHub Secrets
Add these secrets to your GitHub repository:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key | `...` |
| `MONGODB_URL` | MongoDB connection string | `mongodb+srv://...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `SIEVE_API_KEY` | Sieve API key | `...` |
| `TIKTOK_CLIENT_KEY` | TikTok OAuth client key | `...` |
| `TIKTOK_CLIENT_SECRET` | TikTok OAuth secret | `...` |
| `TIKTOK_REDIRECT_URI` | TikTok redirect URI | `https://your-domain.com/tiktok/callback/` |
| `JWT_SECRET_KEY` | JWT signing secret | `your-super-secret-key` |

### Step 4: Deploy Application
```bash
# Push to main branch to trigger deployment
git add .
git commit -m "Deploy to AWS"
git push origin main
```

## 📁 Project Structure

```
clip-it-backend/
├── .github/
│   └── workflows/
│       └── deploy-aws.yml          # GitHub Actions workflow
├── infrastructure/
│   ├── app.py                      # CDK infrastructure code
│   ├── cdk.json                    # CDK configuration
│   └── requirements.txt            # Python dependencies for CDK
├── scripts/
│   ├── deploy.sh                   # Deployment script
│   └── destroy.sh                  # Destruction script
├── Dockerfile.web                  # Web application container
├── Dockerfile.worker               # Worker application container
├── app.py                          # FastAPI application
├── worker.py                       # Background worker
├── jobs.py                         # Job queue management
└── requirements.txt                # Python dependencies
```

## 🔧 Infrastructure Components

### ECS Fargate Services
- **Web Service**: Handles API requests
  - CPU: 1 vCPU
  - Memory: 2 GB
  - Auto-scaling: 1-10 instances
- **Worker Service**: Processes background jobs
  - CPU: 2 vCPU
  - Memory: 4 GB
  - Auto-scaling: 1-5 instances

### ElastiCache Redis
- **Instance Type**: cache.t3.micro
- **Engine**: Redis
- **Purpose**: Job queue and caching

### S3 Storage
- **Purpose**: Video and clip storage
- **Encryption**: S3-managed encryption
- **Versioning**: Enabled

### Application Load Balancer
- **Type**: Application Load Balancer
- **Health Checks**: `/health` endpoint
- **Auto-scaling**: Based on CPU utilization

## 🔄 Deployment Process

### GitHub Actions Workflow
The deployment is automated through GitHub Actions:

1. **Build**: Creates Docker images for web and worker
2. **Push**: Pushes images to ECR
3. **Deploy**: Updates ECS services with new images
4. **Health Check**: Verifies deployment success

### Manual Deployment
```bash
# Build and push images manually
docker build -f Dockerfile.web -t clip-it-web .
docker build -f Dockerfile.worker -t clip-it-worker .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag clip-it-web:latest <account>.dkr.ecr.us-east-1.amazonaws.com/clip-it-web:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/clip-it-web:latest
```

## 📊 Monitoring and Logs

### CloudWatch Logs
- **Log Group**: `/ecs/clip-it`
- **Retention**: 7 days
- **Streams**: Separate streams for web and worker

### Health Checks
- **Web Service**: `GET /health`
- **Worker Service**: Redis connection check
- **Load Balancer**: HTTP health checks

### Metrics
- CPU utilization
- Memory usage
- Request count
- Error rates

## 💰 Cost Estimation

### Monthly Costs (US East 1)

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

## 🔧 Configuration

### Environment Variables

#### Web Service
```bash
REDIS_URL=redis://your-redis-endpoint:6379
MONGODB_URL=mongodb+srv://...
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
OPENAI_API_KEY=sk-...
SIEVE_API_KEY=...
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...
TIKTOK_REDIRECT_URI=https://...
JWT_SECRET_KEY=...
```

#### Worker Service
```bash
REDIS_URL=redis://your-redis-endpoint:6379
MONGODB_URL=mongodb+srv://...
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
OPENAI_API_KEY=sk-...
SIEVE_API_KEY=...
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Deployment Fails
```bash
# Check ECS service status
aws ecs describe-services --cluster clip-it-cluster --services clip-it-web-service

# Check task logs
aws logs get-log-events --log-group-name /ecs/clip-it --log-stream-name web/...
```

#### 2. Redis Connection Issues
```bash
# Check Redis cluster status
aws elasticache describe-cache-clusters --cache-cluster-id clip-it-redis-cluster
```

#### 3. S3 Access Issues
```bash
# Check S3 bucket policy
aws s3api get-bucket-policy --bucket your-bucket-name
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
