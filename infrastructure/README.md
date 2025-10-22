# Clip-It AWS Infrastructure

This directory contains the AWS infrastructure code using AWS CDK (Cloud Development Kit).

## 📁 Files

- `app.py` - Main CDK application defining all AWS resources
- `cdk.json` - CDK configuration file
- `requirements.txt` - Python dependencies for CDK
- `README.md` - This file

## 🏗️ Infrastructure Components

### ECS Fargate
- **Web Service**: FastAPI application
- **Worker Service**: Background job processing
- **Auto-scaling**: Based on CPU utilization

### ElastiCache Redis
- **Purpose**: Job queue and caching
- **Instance**: cache.t3.micro
- **Engine**: Redis

### S3 Storage
- **Purpose**: Video and clip storage
- **Encryption**: S3-managed encryption
- **Versioning**: Enabled

### Application Load Balancer
- **Type**: Application Load Balancer
- **Health Checks**: `/health` endpoint
- **Auto-scaling**: Integrated with ECS

### VPC and Networking
- **VPC**: Custom VPC with public/private subnets
- **Security Groups**: Configured for ECS, Redis, and ALB
- **NAT Gateway**: For private subnet internet access

## 🚀 Quick Start

### Prerequisites
- AWS CLI configured
- Node.js installed
- Python 3.7+ installed
- AWS CDK installed

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy infrastructure
cdk deploy
```

### Destroy
```bash
# Destroy all resources
cdk destroy
```

## 🔧 Configuration

### Environment Variables
The infrastructure uses the following environment variables:
- `AWS_REGION` - AWS region (default: us-east-1)
- `AWS_ACCOUNT` - AWS account ID (auto-detected)

### Customization
To modify the infrastructure:
1. Edit `app.py`
2. Run `cdk deploy` to apply changes

## 📊 Outputs

After deployment, the following outputs are available:
- **LoadBalancerURL**: Application URL
- **S3BucketName**: S3 bucket for file storage
- **RedisEndpoint**: Redis cluster endpoint

## 🔍 Monitoring

### CloudWatch Logs
- **Log Group**: `/ecs/clip-it`
- **Retention**: 7 days
- **Streams**: Separate for web and worker

### Health Checks
- **Web Service**: `GET /health`
- **Worker Service**: Redis connection check
- **Load Balancer**: HTTP health checks

## 💰 Cost Optimization

### Recommendations
1. **Use Spot Instances** for workers (60-90% savings)
2. **S3 Intelligent Tiering** for storage optimization
3. **Reserved Instances** for predictable workloads
4. **CloudWatch Logs Retention** (reduce from 7 days)

### Estimated Monthly Costs
- **Low Usage**: ~$295/month
- **Medium Usage**: ~$730/month
- **High Usage**: ~$1,210/month

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
# Check CloudFormation stack
aws cloudformation describe-stacks --stack-name ClipItStack

# Check ECS service status
aws ecs describe-services --cluster clip-it-cluster
```

#### 3. Redis Connection Issues
```bash
# Check Redis cluster status
aws elasticache describe-cache-clusters
```

### Debug Commands
```bash
# View CDK diff
cdk diff

# View synthesized CloudFormation
cdk synth

# Check deployment status
cdk list
```

## 📚 Additional Resources

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [ElastiCache Documentation](https://docs.aws.amazon.com/elasticache/)
- [S3 Documentation](https://docs.aws.amazon.com/s3/)

---

**Happy Deploying! 🚀**
