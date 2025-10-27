#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import Stack
from aws_cdk import aws_autoscaling as autoscaling
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
            nat_gateways=1,
            enable_dns_hostnames=True,
            enable_dns_support=True
        )

        # Create S3 bucket for file storage
        s3_bucket = s3.Bucket(
            self, "ClipItStorage",
            bucket_name=f"clip-it-storage-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=cdk.RemovalPolicy.RETAIN
        )

        # Create ElastiCache Redis cluster
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisSubnetGroup",
            description="Subnet group for Redis cluster",
            subnet_ids=[subnet.subnet_id for subnet in vpc.private_subnets]
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

        # Create ECR repositories
        web_repo = ecr.Repository(
            self, "WebRepository",
            repository_name="clip-it-web",
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10,
                    rule_priority=1
                )
            ]
        )

        worker_repo = ecr.Repository(
            self, "WorkerRepository",
            repository_name="clip-it-worker",
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10,
                    rule_priority=1
                )
            ]
        )

        # Create task execution role
        task_execution_role = iam.Role(
            self, "TaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
            ]
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

        # Environment variables
        env_vars = {
            "REDIS_URL": f"redis://{redis_cluster.attr_redis_endpoint_address}:6379",
            "S3_BUCKET": s3_bucket.bucket_name,
            "AWS_REGION": self.region
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
            port_mappings=[ecs.PortMapping(container_port=8000)]
        )

        # Worker service task definition
        worker_task_definition = ecs.FargateTaskDefinition(
            self, "WorkerTaskDefinition",
            cpu=2048,
            memory_limit_mib=4096,
            execution_role=task_execution_role,
            task_role=task_role,
            family="clip-it-worker-task"
        )

        worker_container = worker_task_definition.add_container(
            "clip-it-worker",
            image=ecs.ContainerImage.from_ecr_repository(worker_repo, "latest"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="worker",
                log_group=log_group
            ),
            environment=env_vars
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
                timeout=cdk.Duration.seconds(5),
                interval=cdk.Duration.seconds(30)
            )
        )

        # Create listener
        listener = load_balancer.add_listener(
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

        worker_service = ecs.FargateService(
            self, "WorkerService",
            cluster=cluster,
            task_definition=worker_task_definition,
            desired_count=0,  # Start with 0 tasks to avoid immediate failures
            service_name="clip-it-worker-service",
            assign_public_ip=True,
            security_groups=[ecs_security_group],
            enable_execute_command=True  # Enable debugging
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

        # Create auto-scaling for worker service
        worker_scaling = worker_service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=5
        )

        worker_scaling.scale_on_cpu_utilization(
            "WorkerCpuScaling",
            target_utilization_percent=80
        )

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
        cdk.CfnOutput(
            self, "LoadBalancerURL",
            value=f"http://{load_balancer.load_balancer_dns_name}",
            description="Application Load Balancer URL"
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
