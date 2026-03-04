import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
import time

import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self):
        # Connect to Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
            
            # Worker identification for distributed locking
            self.worker_id = f"worker-{uuid.uuid4()}"
            logger.info(f"Worker ID: {self.worker_id}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage for local development
            self.redis_client = None
            self._memory_storage = {}
            self.worker_id = "local-worker"

    def add_job(self, job_type: str, job_data: Dict[str, Any]) -> str:
        """Add a job to the queue"""
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": job_type,
            "data": job_data,
            "status": "queued",
            "progress": "0",
            "message": "Job queued",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        if self.redis_client:
            # Store job data (convert to strings for Redis)
            for k, v in job.items():
                value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                self.redis_client.hset(f"job:{job_id}", k, value)
            # Add to queue
            self.redis_client.lpush("job_queue", job_id)
        else:
            # Fallback to memory
            self._memory_storage[job_id] = job

        logger.info(f"Added job {job_id} of type {job_type} to queue")

        # Trigger GPU worker scale-up (no-op if already running or disabled)
        self.scale_up_gpu_worker()

        return job_id

    def scale_up_gpu_worker(self):
        """Scale up the GPU worker ECS service + ASG when a new job arrives.

        Only runs if AUTO_SCALE_ENABLED=true (so local dev is unaffected).
        If the service is already running (desiredCount > 0) this is a no-op.

        Race-condition handling:
        If the worker is currently in the middle of a scale-down (indicated by
        the 'worker:scaling_down' Redis key), we cancel that scale-down by
        deleting the key.  scale_down_self() checks for the key after its 15s
        wait and will abort termination if the key is gone, then restores
        desiredCount=1.  So we don't need to touch ECS/ASG here — the worker
        process stays alive and picks up the queued job itself.
        """
        from config import settings
        if not settings.AUTO_SCALE_ENABLED:
            return

        try:
            import boto3

            # If the worker is mid-shutdown, cancel it by deleting the lock key.
            # scale_down_self() will see the key is gone, abort termination, and
            # restore ECS desiredCount=1 on its own — no AWS API calls needed here.
            if self.redis_client and self.redis_client.exists("worker:scaling_down"):
                self.redis_client.delete("worker:scaling_down")
                logger.info(
                    "New job arrived while worker was shutting down — "
                    "scale-down cancelled (worker:scaling_down lock cleared). "
                    "Worker will stay alive and process the job."
                )
                return

            # Use ECS_REGION (plain env var) not AWS_REGION (may come from Secrets
            # Manager with a wrong value like us-east-2 which causes ClusterNotFoundException)
            region = settings.ECS_REGION
            cluster_name = settings.ECS_CLUSTER_NAME
            worker_service_name = settings.WORKER_SERVICE_NAME
            asg_name = settings.WORKER_ASG_NAME

            ecs_client = boto3.client("ecs", region_name=region)

            # Check current desired count
            response = ecs_client.describe_services(
                cluster=cluster_name,
                services=[worker_service_name]
            )

            if not response.get("services"):
                logger.warning(f"GPU worker service '{worker_service_name}' not found, skipping scale-up")
                return

            current_desired = response["services"][0]["desiredCount"]

            if current_desired == 0:
                logger.info("GPU worker is stopped (desiredCount=0), scaling up...")

                # Step 1: Scale the ASG to 1 so an EC2 instance is available
                if asg_name:
                    asg_client = boto3.client("autoscaling", region_name=region)
                    asg_client.set_desired_capacity(
                        AutoScalingGroupName=asg_name,
                        DesiredCapacity=1,
                        HonorCooldown=False
                    )
                    logger.info(f"Scaled ASG '{asg_name}' desired capacity to 1")

                # Step 2: Scale ECS service to 1
                ecs_client.update_service(
                    cluster=cluster_name,
                    service=worker_service_name,
                    desiredCount=1
                )
                logger.info("GPU worker ECS service scaled up to 1 task")
            else:
                logger.debug(f"GPU worker already active (desiredCount={current_desired}), no scale-up needed")

        except Exception as e:
            # Scale-up failure must not block the job — just log it
            logger.error(f"Failed to scale up GPU worker: {e}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details by ID"""
        if self.redis_client:
            job_data = self.redis_client.hgetall(f"job:{job_id}")
            return job_data if job_data else None
        else:
            return self._memory_storage.get(job_id)

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        """Update job status and data"""
        updates["updated_at"] = datetime.now().isoformat()

        if self.redis_client:
            for k, v in updates.items():
                value = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                self.redis_client.hset(f"job:{job_id}", k, value)
        else:
            if job_id in self._memory_storage:
                self._memory_storage[job_id].update(updates)

        logger.info(f"Updated job {job_id}: {updates}")

    def get_next_job(self) -> Optional[str]:
        """Get the next job from the queue (blocking) with distributed locking"""
        if self.redis_client:
            result = self.redis_client.brpop("job_queue", timeout=30)
            if not result:
                return None
                
            job_id = result[1]
            
            # Check if job still exists and hasn't been cancelled
            job = self.get_job(job_id)
            if not job:
                logger.info(f"Job {job_id} no longer exists, skipping")
                return None
            
            # Check if job was cancelled
            job_status = job.get("status", "")
            if job_status in ["cancelled", "error", "completed"]:
                logger.info(f"Job {job_id} already {job_status}, deleting from Redis")
                self.redis_client.delete(f"job:{job_id}")
                self.redis_client.delete(f"job:lock:{job_id}")
                return None
            
            # Try to acquire lock for this job (prevents duplicate processing)
            lock_key = f"job:lock:{job_id}"
            lock_acquired = self.redis_client.set(
                lock_key,
                self.worker_id,
                nx=True,  # Only set if not exists
                ex=3600   # Lock expires in 1 hour (safety mechanism)
            )
            
            if lock_acquired:
                logger.info(f"Worker {self.worker_id} acquired lock for job {job_id}")
                # Mark job as processing
                self.update_job(job_id, {
                    "status": "processing",
                    "worker_id": self.worker_id,
                    "processing_started_at": datetime.now().isoformat()
                })
                return job_id
            else:
                # Job is already locked by another worker - do NOT requeue!
                # The other worker is already processing it.
                lock_owner = self.redis_client.get(lock_key)
                logger.debug(f"Job {job_id} is being processed by worker {lock_owner}, skipping")
                return None
        else:
            # For memory storage, just return None (no queuing in development)
            return None

    def release_job_lock(self, job_id: str):
        """Release the distributed lock for a job"""
        if self.redis_client:
            lock_key = f"job:lock:{job_id}"
            # Only delete the lock if we own it
            current_owner = self.redis_client.get(lock_key)
            if current_owner == self.worker_id:
                self.redis_client.delete(lock_key)
                logger.info(f"Worker {self.worker_id} released lock for job {job_id}")
            elif current_owner:
                logger.warning(f"Cannot release lock for job {job_id} - owned by {current_owner}")
            else:
                logger.debug(f"No lock to release for job {job_id}")
    
    def requeue_job(self, job_id: str):
        """Requeue a job (e.g., on failure or interruption) and release lock"""
        if self.redis_client:
            # Update status to 'rescheduling' for user feedback
            self.update_job(job_id, {
                "status": "rescheduling",
                "message": "Resource reclamation in progress. Rescheduling job..."
            })
            self.release_job_lock(job_id)
            self.redis_client.lpush("job_queue", job_id)
            logger.info(f"Job {job_id} requeued for retry")

    def delete_job(self, job_id: str):
        """Delete a completed/cancelled job from Redis to prevent accumulation"""
        if self.redis_client:
            # Delete the job hash
            self.redis_client.delete(f"job:{job_id}")
            # Delete any lock (just in case)
            self.redis_client.delete(f"job:lock:{job_id}")
            logger.info(f"Job {job_id} deleted from Redis")

    def recover_stuck_jobs(self):
        """Find jobs in 'processing' state and put them back in queue (for worker startup)"""
        if self.redis_client:
            logger.info("Checking for stuck jobs to recover...")
            all_jobs = self.get_all_jobs()
            for job_id, job in all_jobs.items():
                if job.get("status") == "processing" or job.get("status") == "rescheduling":
                    logger.info(f"Recovering stuck job {job_id} (status: {job.get('status')})")
                    self.requeue_job(job_id)


    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all jobs (for debugging)"""
        if self.redis_client:
            jobs = {}
            for key in self.redis_client.scan_iter(match="job:*"):
                job_id = key.split(":", 1)[1]
                jobs[job_id] = self.redis_client.hgetall(key)
            return jobs
        else:
            return self._memory_storage.copy()


# Global job queue instance
job_queue = JobQueue()
