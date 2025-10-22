import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

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
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage for local development
            self.redis_client = None
            self._memory_storage = {}

    def add_job(self, job_type: str, job_data: Dict[str, Any]) -> str:
        """Add a job to the queue"""
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": job_type,
            "data": job_data,
            "status": "queued",
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
        """Get the next job from the queue (blocking)"""
        if self.redis_client:
            result = self.redis_client.brpop("job_queue", timeout=30)
            return result[1] if result else None
        else:
            # For memory storage, just return None (no queuing in development)
            return None

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
