"""
TikTok OAuth token management service.
Handles storage, retrieval, and refresh of TikTok access tokens.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from database.connection import get_database

logger = logging.getLogger(__name__)


class TikTokTokenService:
    """Service for managing TikTok OAuth tokens"""
    
    def __init__(self):
        self.db = None
        self.collection = None
    
    def _get_collection(self):
        if self.db is None:
            self.db = get_database()
            self.collection = self.db["tiktok_tokens"]
            # Ensure unique index on user_id so each user has at most one token record
            try:
                self.collection.create_index("user_id", unique=True)
            except Exception as e:
                logger.warning(f"Could not ensure unique index on tiktok_tokens.user_id: {e}")
        return self.collection
    
    async def save_tokens(self, user_id: str, tokens: dict) -> None:
        """
        Save TikTok OAuth tokens for a user.
        
        Args:
            user_id: User identifier
            tokens: Dictionary containing TikTok tokens
                   Expected keys: open_id, access_token, refresh_token, scope, expires_at
        """
        try:
            # Normalize user_id to string for storage
            if not isinstance(user_id, str):
                user_id = str(user_id)

            if not user_id:
                raise ValueError("Missing user_id for saving TikTok tokens")

            required = ["access_token", "refresh_token", "expires_at"]
            for key in required:
                if tokens.get(key) in (None, ""):
                    raise ValueError(f"Missing required token field: {key}")
            
            # Prepare token document
            token_doc = {
                "user_id": user_id,
                "open_id": tokens.get("open_id"),
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "scope": tokens.get("scope"),
                "expires_at": tokens.get("expires_at"),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # Upsert tokens (update if exists, insert if not)
            collection = self._get_collection()
            result = collection.update_one(
                {"user_id": user_id},
                {"$set": token_doc},
                upsert=True
            )
            
            logger.info(f"TikTok tokens saved for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error saving TikTok tokens for user {user_id}: {str(e)}")
            raise
    
    async def get_tokens(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve TikTok tokens for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing tokens or None if not found
        """
        try:
            # Normalize user_id to string for lookup
            if not isinstance(user_id, str):
                user_id = str(user_id)
            
            collection = self._get_collection()
            token_doc = collection.find_one({"user_id": user_id})
            
            if not token_doc:
                return None
            
            # Convert ids to strings for JSON serialization if present
            if "user_id" in token_doc and not isinstance(token_doc["user_id"], str):
                token_doc["user_id"] = str(token_doc["user_id"])
            if "_id" in token_doc and not isinstance(token_doc["_id"], str):
                token_doc["_id"] = str(token_doc["_id"])
            
            return token_doc
            
        except Exception as e:
            logger.error(f"Error retrieving TikTok tokens for user {user_id}: {str(e)}")
            return None
    
    async def update_tokens(self, user_id: str, **fields) -> None:
        """
        Update specific fields in TikTok tokens.
        
        Args:
            user_id: User identifier
            **fields: Fields to update (e.g., access_token, expires_at, etc.)
        """
        try:
            # Normalize user_id to string
            if not isinstance(user_id, str):
                user_id = str(user_id)
            
            # Add updated_at timestamp
            fields["updated_at"] = datetime.now(timezone.utc)
            
            collection = self._get_collection()
            result = collection.update_one(
                {"user_id": user_id},
                {"$set": fields}
            )
            
            if result.matched_count == 0:
                logger.warning(f"No TikTok tokens found for user {user_id} to update")
            else:
                logger.info(f"TikTok tokens updated for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error updating TikTok tokens for user {user_id}: {str(e)}")
            raise
    
    async def delete_tokens(self, user_id: str) -> None:
        """
        Delete TikTok tokens for a user.
        
        Args:
            user_id: User identifier
        """
        try:
            # Normalize user_id to string
            if not isinstance(user_id, str):
                user_id = str(user_id)
            
            collection = self._get_collection()
            result = collection.delete_one({"user_id": user_id})
            
            if result.deleted_count > 0:
                logger.info(f"TikTok tokens deleted for user {user_id}")
            else:
                logger.warning(f"No TikTok tokens found for user {user_id} to delete")
                
        except Exception as e:
            logger.error(f"Error deleting TikTok tokens for user {user_id}: {str(e)}")
            raise
    
    async def is_token_expired(self, user_id: str) -> bool:
        """
        Check if TikTok access token is expired.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if token is expired or not found, False otherwise
        """
        try:
            tokens = await self.get_tokens(user_id)
            if not tokens or not tokens.get("expires_at"):
                return True
            
            # Check if token expires within the next 5 minutes
            expires_at = tokens["expires_at"]
            if isinstance(expires_at, int):
                expires_at = datetime.fromtimestamp(expires_at, tz=timezone.utc)
            
            now = datetime.now(timezone.utc)
            # Add 5 minutes buffer to refresh before actual expiration
            buffer_time = now.replace(tzinfo=timezone.utc) + timedelta(minutes=5)
            return expires_at <= buffer_time
            
        except Exception as e:
            logger.error(f"Error checking token expiration for user {user_id}: {str(e)}")
            return True


# Global instance
tiktok_token_service = TikTokTokenService()
