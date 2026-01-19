#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticache as elasticache
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_elasticloadbalancingv2_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_ssm as ssm
from constructs import Construct

class ClipItStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create VPC
        vpc = ec2.Vpc(
            self, "ClipItVPC",
            max_azs=2,
            nat_gateways=0,  # Not needed since GPU worker uses PUBLIC subnet
            enable_dns_hostnames=True,
            enable_dns_support=True,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,  # No NAT needed
                    cidr_mask=24
                )
            ]
        )

        # Create S3 bucket for file storage
        # Import existing S3 bucket (created outside stack or from previous deployment)
        bucket_name = f"clip-it-storage-{self.account}-{self.region}"
        s3_bucket = s3.Bucket.from_bucket_name(
            self, "ClipItStorage",
            bucket_name=bucket_name
        )

        # Create ElastiCache Redis cluster
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisSubnetGroup",
            description="Subnet group for Redis cluster",
            subnet_ids=[subnet.subnet_id for subnet in vpc.isolated_subnets]
        )

        redis_security_group = ec2.SecurityGroup(
            self, "RedisSecurityGroup",
            vpc=vpc,
            description="Security group for Redis",
            allow_all_outbound=False
        )

        redis_security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(6379),
            description="Allow Redis access from VPC"
        )

        redis_cluster = elasticache.CfnCacheCluster(
            self, "RedisCluster",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_nodes=1,
            vpc_security_group_ids=[redis_security_group.security_group_id],
            cache_subnet_group_name=redis_subnet_group.ref
        )

        # Create ECS Cluster
        cluster = ecs.Cluster(
            self, "ClipItCluster",
            vpc=vpc,
            cluster_name="clip-it-cluster"
        )

        # Import existing ECR repositories (created outside stack or from previous deployment)
        web_repo = ecr.Repository.from_repository_name(
            self, "WebRepository",
            repository_name="clip-it-web"
        )

        worker_repo = ecr.Repository.from_repository_name(
            self, "WorkerRepository",
            repository_name="clip-it-worker"
        )

        # ========================================
        # EC2 GPU INFRASTRUCTURE FOR WORKERS
        # ========================================

        # Create IAM role for EC2 instances (separate from launch template)
        ec2_instance_role = iam.Role(
            self, "EC2InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )

        # Get GPU-optimized AMI
        gpu_ami = ecs.EcsOptimizedImage.amazon_linux2(
            hardware_type=ecs.AmiHardwareType.GPU
        )

        # Security group for GPU worker instances
        worker_sg = ec2.SecurityGroup(
            self, "WorkerEC2SecurityGroup",
            vpc=vpc,
            description="Security group for GPU EC2 worker instances",
            allow_all_outbound=True
        )

        # Launch Template for GPU instances (On-Demand)

        spot_launch_template = ec2.LaunchTemplate(
            self, "GPUSpotLaunchTemplate",
            instance_type=ec2.InstanceType("g4dn.xlarge"),
            machine_image=gpu_ami,
            role=ec2_instance_role,
            user_data=ec2.UserData.for_linux(),
            require_imdsv2=True,
            # Associate public IP for ECS connectivity
            associate_public_ip_address=True,
            security_group=worker_sg
        )

        spot_launch_template.user_data.add_commands(
            "exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1",
            "set -x",
            "echo '=== GPU Worker Instance Setup Started ==='",
            "date",
            f"echo ECS_CLUSTER={cluster.cluster_name} >> /etc/ecs/ecs.config",
            "echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config",
            "echo ECS_AVAILABLE_LOGGING_DRIVERS='[\"json-file\",\"awslogs\"]' >> /etc/ecs/ecs.config",
            "echo '=== ECS Configuration ==='",
            "cat /etc/ecs/ecs.config",
            "echo '=== GPU Check ==='",
            "nvidia-smi || echo 'nvidia-smi failed'",
            "echo '=== Restarting ECS Agent ==='",
            "systemctl restart ecs",
            "sleep 5",
            "systemctl status ecs",
            "echo '=== Setup Complete ==='",
            "date"
        )

        # Auto Scaling Group for 1 Spot Worker
        gpu_asg = autoscaling.AutoScalingGroup(
            self, "GPUWorkerASG",
            vpc=vpc,
            launch_template=spot_launch_template,
            min_capacity=0,  # Allow scaling down to 0 when not processing videos
            max_capacity=1,
            desired_capacity=0,  # Start at 0 to save costs when not in use
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
        )

        # Capacity Provider for GPU ASG
        gpu_capacity_provider = ecs.AsgCapacityProvider(
            self, "GPUCapacityProvider",
            auto_scaling_group=gpu_asg,
            enable_managed_termination_protection=False, # Keeping this disabled to avoid complexity with scale-to-zero
            enable_managed_scaling=True,
            target_capacity_percent=100
        )
        cluster.add_asg_capacity_provider(gpu_capacity_provider)


        # Create task execution role
        task_execution_role = iam.Role(
            self, "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
        )
        
        # Add SSM permissions for reading secrets
        task_execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                    "ssm:GetParametersByPath"
                ],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/clip-it/*"
                ]
            )
        )
        
        # Add KMS permissions for decrypting SecureString parameters
        task_execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "kms:Decrypt"
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "kms:ViaService": f"ssm.{self.region}.amazonaws.com"
                    }
                }
            )
        )

        # Create task role
        task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )

        # Add S3 permissions
        task_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                resources=[
                    s3_bucket.bucket_arn,
                    f"{s3_bucket.bucket_arn}/*"
                ]
            )
        )

        # Create CloudWatch log group
        log_group = logs.LogGroup(
            self, "ClipItLogGroup",
            log_group_name="/ecs/clip-it",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=cdk.RemovalPolicy.DESTROY
        )

        # Environment variables (non-sensitive)
        # Note: S3_BUCKET and AWS_REGION can come from SSM if needed, but we'll use CDK values as defaults
        env_vars = {
            "REDIS_URL": f"redis://{redis_cluster.attr_redis_endpoint_address}:6379",
            "MONGODB_DB_NAME": "clip_it_db",
            "JWT_ALGORITHM": "HS256",
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "180",
            "JWT_REFRESH_TOKEN_EXPIRE_DAYS": "10",
            "TIKTOK_SCOPES": "user.info.basic,user.info.profile,video.publish,video.upload",
            "TIKTOK_API_BASE": "https://open.tiktokapis.com/v2",
            "TIKTOK_AUTH_BASE": "https://www.tiktok.com/v2",
            "MIN_CLIP_DURATION": "15",
            "PREFERRED_CLIP_DURATION": "180",
            "MAX_CLIPS_PER_EPISODE": "10",
            "OUTPUT_WIDTH": "1080",
            "OUTPUT_HEIGHT": "1920",
            "SPEAKER_DIARIZATION_ENABLED": "false"
        }
        
        # Secrets from SSM (sensitive values)
        # Using from_secure_string_parameter_attributes for SecureString parameters
        # These parameters should exist in SSM (created via scripts/setup-secrets.sh)
        secrets = {
            "OPENAI_API_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "OpenAIAPIKeyRef",
                    parameter_name="/clip-it/openai-api-key"
                )
            ),
            "SIEVE_API_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "SieveAPIKeyRef",
                    parameter_name="/clip-it/sieve-api-key"
                )
            ),
            "RAPIDAPI_YOUTUBE_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "RapidAPIYouTubeKeyRef",
                    parameter_name="/clip-it/rapidapi-youtube-key"
                )
            ),
            "MONGODB_URL": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "MongoDBURLRef",
                    parameter_name="/clip-it/mongodb-url"
                )
            ),
            "JWT_SECRET_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "JWTSecretKeyRef",
                    parameter_name="/clip-it/jwt-secret-key"
                )
            ),
            "TIKTOK_CLIENT_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "TikTokClientKeyRef",
                    parameter_name="/clip-it/tiktok-client-key"
                )
            ),
            "TIKTOK_CLIENT_SECRET": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "TikTokClientSecretRef",
                    parameter_name="/clip-it/tiktok-client-secret"
                )
            ),
            "TIKTOK_REDIRECT_URI": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "TikTokRedirectURIRef",
                    parameter_name="/clip-it/tiktok-redirect-uri"
                )
            ),
            "TIKTOK_VERIFICATION_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "TikTokVerificationKeyRef",
                    parameter_name="/clip-it/tiktok-verification-key"
                )
            ),
            "PROXY_BASE_URL": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "ProxyBaseURLRef",
                    parameter_name="/clip-it/proxy-base-url"
                )
            ),
            "AWS_ACCESS_KEY_ID": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "AWSAccessKeyIDRef",
                    parameter_name="/clip-it/aws-access-key-id"
                )
            ),
            "AWS_SECRET_ACCESS_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "AWSSecretAccessKeyRef",
                    parameter_name="/clip-it/aws-secret-access-key"
                )
            ),
            "AWS_REGION": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "AWSRegionRef",
                    parameter_name="/clip-it/aws-region"
                )
            ),
            "S3_BUCKET": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "S3BucketRef",
                    parameter_name="/clip-it/s3-bucket"
                )
            ),
            "HF_TOKEN": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "HfTokenRef",
                    parameter_name="/clip-it/hf-token"
                )
            ),
            "HUGGINGFACE_TOKEN": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "HuggingfaceTokenRef",
                    parameter_name="/clip-it/huggingface-token"
                )
            ),
            "ASSEMBLYAI_API_KEY": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "AssemblyAIAPIKeyRef",
                    parameter_name="/clip-it/assemblyai-api-key"
                )
            ),
            "YOUTUBE_CLIENT_ID": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "YouTubeClientIDRef",
                    parameter_name="/clip-it/youtube-client-id"
                )
            ),
            "YOUTUBE_CLIENT_SECRET": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "YouTubeClientSecretRef",
                    parameter_name="/clip-it/youtube-client-secret"
                )
            ),
            "YOUTUBE_REDIRECT_URI": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "YouTubeRedirectURIRef",
                    parameter_name="/clip-it/youtube-redirect-uri"
                )
            ),
            "INSTA_APP_ID": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "InstaAppIDRef",
                    parameter_name="/clip-it/insta-app-id"
                )
            ),
            "INSTA_APP_SECRET": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "InstaAppSecretRef",
                    parameter_name="/clip-it/insta-app-secret"
                )
            ),
            "INSTA_REDIRECT_URI": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "InstaRedirectURIRef",
                    parameter_name="/clip-it/insta-redirect-uri"
                )
            ),
            "INSTA_APP_SCOPE": ecs.Secret.from_ssm_parameter(
                ssm.StringParameter.from_secure_string_parameter_attributes(
                    self, "InstaAppScopeRef",
                    parameter_name="/clip-it/insta-app-scope"
                )
            ),
        }

        # Web service task definition
        web_task_definition = ecs.FargateTaskDefinition(
            self, "WebTaskDefinition",
            cpu=1024,
            memory_limit_mib=2048,
            execution_role=task_execution_role,
            task_role=task_role,
            family="clip-it-web-task"
        )

        web_container = web_task_definition.add_container(
            "clip-it-web",
            image=ecs.ContainerImage.from_ecr_repository(web_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="web",
                log_group=log_group
            ),
            environment=env_vars,
            secrets=secrets,
            port_mappings=[ecs.PortMapping(container_port=8000)]
        )

        # Worker service task definition (EC2 for GPU)
        worker_task_definition = ecs.Ec2TaskDefinition(
            self, "WorkerTaskDefinition",
            execution_role=task_execution_role,
            task_role=task_role,
            network_mode=ecs.NetworkMode.AWS_VPC,
            family="clip-it-worker-task"
        )

        worker_container = worker_task_definition.add_container(
            "clip-it-worker",
            image=ecs.ContainerImage.from_ecr_repository(worker_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="worker",
                log_group=log_group
            ),
            environment=env_vars,
            secrets=secrets,
            gpu_count=1,
            memory_limit_mib=8192,
            cpu=2048
        )


        # Create Application Load Balancer
        alb_security_group = ec2.SecurityGroup(
            self, "ALBSecurityGroup",
            vpc=vpc,
            description="Security group for ALB",
            allow_all_outbound=True
        )

        alb_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(80),
            description="Allow HTTP traffic"
        )

        alb_security_group.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443),
            description="Allow HTTPS traffic"
        )

        load_balancer = elbv2.ApplicationLoadBalancer(
            self, "ClipItALB",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_security_group
        )

        # Create target group for web service
        web_target_group = elbv2.ApplicationTargetGroup(
            self, "WebTargetGroup",
            port=8000,
            protocol=elbv2.ApplicationProtocol.HTTP,
            vpc=vpc,
            target_type=elbv2.TargetType.IP,
            health_check=elbv2.HealthCheck(
                enabled=True,
                healthy_http_codes="200",
                path="/health",
                protocol=elbv2.Protocol.HTTP,
                timeout=cdk.Duration.seconds(10),  # Increased from 5 to 10 seconds
                interval=cdk.Duration.seconds(30),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3
            ),
            # Increase deregistration delay to allow in-flight requests to complete
            deregistration_delay=cdk.Duration.seconds(30)
        )

        # Get SSL certificate ARN from environment variable, CDK context, or SSM
        # Priority: 1) Environment variable, 2) CDK context, 3) SSM parameter
        ssl_cert_arn = os.getenv("SSL_CERTIFICATE_ARN") or self.node.try_get_context("ssl_certificate_arn")
        
        # Try to get from SSM if not set (read at synthesis time)
        if not ssl_cert_arn:
            try:
                import subprocess
                result = subprocess.run(
                    ["aws", "ssm", "get-parameter", "--name", "/clip-it/ssl-certificate-arn", "--region", self.region, "--query", "Parameter.Value", "--output", "text"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    ssl_cert_arn = result.stdout.strip()
                    print(f"✅ Found SSL certificate ARN in SSM: {ssl_cert_arn}")
            except Exception as e:
                print(f"⚠️  Could not read SSL certificate from SSM: {e}")
                ssl_cert_arn = None
        
        # Create HTTPS listener if certificate is available
        if ssl_cert_arn:
            # Import the certificate
            certificate = acm.Certificate.from_certificate_arn(
                self, "SSLCertificate",
                certificate_arn=ssl_cert_arn
            )
            
            # Create HTTPS listener (port 443)
            https_listener = load_balancer.add_listener(
                "ClipItHTTPSListener",
                port=443,
                protocol=elbv2.ApplicationProtocol.HTTPS,
                certificates=[certificate],
                default_target_groups=[web_target_group]
            )
            
            # Update HTTP listener to redirect to HTTPS
            # Use same ID as existing listener so CloudFormation updates it instead of creating new one
            http_listener = load_balancer.add_listener(
                "ClipItListener",  # Same ID as before to update existing listener
                port=80,
                default_action=elbv2.ListenerAction.redirect(
                    protocol="HTTPS",
                    port="443",
                    permanent=True
                )
            )
        else:
            # No certificate - use HTTP only (for development)
            # In production, you should always use HTTPS
            http_listener = load_balancer.add_listener(
                "ClipItListener",
                port=80,
                default_target_groups=[web_target_group]
            )

        # Create shared security group for ECS services
        ecs_security_group = self.create_ecs_security_group(vpc)

        # Create ECS services with better configuration
        web_service = ecs.FargateService(
            self, "WebService",
            cluster=cluster,
            task_definition=web_task_definition,
            desired_count=0,  # Start with 0 tasks to avoid immediate failures
            service_name="clip-it-web-service",
            assign_public_ip=True,
            security_groups=[ecs_security_group],
            health_check_grace_period=cdk.Duration.seconds(300),  # 5 minutes
            enable_execute_command=True  # Enable debugging
        )

        worker_service = ecs.Ec2Service(
            self, "WorkerService",
            cluster=cluster,
            task_definition=worker_task_definition,
            desired_count=0,  # Start with 0 tasks (scale up when needed)
            service_name="clip-it-gpu-worker-service",
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(
                    capacity_provider=gpu_capacity_provider.capacity_provider_name,
                    weight=1
                )
            ],
            security_groups=[ecs_security_group],
            enable_execute_command=True,
            # Allow stopping old task before new one starts (required for single GPU)
            min_healthy_percent=0,
            max_healthy_percent=200
        )


        # Register web service with target group
        web_service.attach_to_application_target_group(web_target_group)

        # Create auto-scaling for web service
        web_scaling = web_service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10
        )

        web_scaling.scale_on_cpu_utilization(
            "WebCpuScaling",
            target_utilization_percent=70
        )

        # No auto-scaling for GPU worker (fixed to 1)
        # We manually keep it at 1 for cost management as per request


        # Store important values in SSM Parameter Store
        ssm.StringParameter(
            self, "S3BucketNameParam",
            parameter_name="/clip-it/s3-bucket-name",
            string_value=s3_bucket.bucket_name
        )

        ssm.StringParameter(
            self, "RedisEndpointParam",
            parameter_name="/clip-it/redis-endpoint",
            string_value=redis_cluster.attr_redis_endpoint_address
        )

        ssm.StringParameter(
            self, "LoadBalancerDNSParam",
            parameter_name="/clip-it/load-balancer-dns",
            string_value=load_balancer.load_balancer_dns_name
        )

        # Outputs
        if ssl_cert_arn:
            cdk.CfnOutput(
                self, "LoadBalancerURL",
                value=f"https://{load_balancer.load_balancer_dns_name}",
                description="Application Load Balancer URL (HTTPS)"
            )
            cdk.CfnOutput(
                self, "LoadBalancerHTTPURL",
                value=f"http://{load_balancer.load_balancer_dns_name}",
                description="Application Load Balancer URL (HTTP - redirects to HTTPS)"
            )
        else:
            cdk.CfnOutput(
                self, "LoadBalancerURL",
                value=f"http://{load_balancer.load_balancer_dns_name}",
                description="Application Load Balancer URL (HTTP only - set SSL_CERTIFICATE_ARN for HTTPS)"
            )

        cdk.CfnOutput(
            self, "S3BucketName",
            value=s3_bucket.bucket_name,
            description="S3 Bucket for file storage"
        )

        cdk.CfnOutput(
            self, "RedisEndpoint",
            value=redis_cluster.attr_redis_endpoint_address,
            description="Redis cluster endpoint"
        )

    def create_ecs_security_group(self, vpc):
        """Create security group for ECS tasks"""
        security_group = ec2.SecurityGroup(
            self, "ECSSecurityGroup",
            vpc=vpc,
            description="Security group for ECS tasks",
            allow_all_outbound=True
        )

        security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(8000),
            description="Allow HTTP from ALB"
        )

        security_group.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.tcp(6379),
            description="Allow Redis access"
        )

        return security_group

app = cdk.App()
ClipItStack(app, "ClipItStack")
app.synth()
