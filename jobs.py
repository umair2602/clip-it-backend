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
        return job_id

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
                # Another worker got this job, put it back and try again
                lock_owner = self.redis_client.get(lock_key)
                logger.warning(f"Job {job_id} already locked by worker {lock_owner}, requeueing...")
                self.redis_client.lpush("job_queue", job_id)
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
