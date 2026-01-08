
"""
YouTube OAuth token management service.
Handles storage, retrieval, and refresh of YouTube access tokens.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from database.connection import get_database
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)


class YouTubeTokenService:
    """Service for managing YouTube OAuth tokens"""

    def __init__(self):
        self.db = None
        self.collection = None

    def _get_collection(self):
        if self.db is None:
            self.db = get_database()
            self.collection = self.db["youtube_tokens"]
            try:
                self.collection.create_index("user_id", unique=True)
            except Exception as e:
                logger.warning(f"Could not ensure unique index on youtube_tokens.user_id: {e}")
        return self.collection

    async def save_tokens(self, user_id: str, credentials) -> None:
        """
        Save YouTube OAuth tokens for a user.
        """
        try:
            if not isinstance(user_id, str):
                user_id = str(user_id)

            if not user_id:
                raise ValueError("Missing user_id for saving YouTube tokens")

            token_doc = {
                "user_id": user_id,
                "token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_uri": credentials.token_uri,
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "scopes": credentials.scopes,
                "expiry": credentials.expiry,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }

            collection = self._get_collection()
            collection.update_one(
                {"user_id": user_id},
                {"$set": token_doc},
                upsert=True
            )

            logger.info(f"YouTube tokens saved for user {user_id}")

        except Exception as e:
            logger.error(f"Error saving YouTube tokens for user {user_id}: {str(e)}")
            raise

    async def get_tokens(self, user_id: str) -> Optional[Credentials]:
        """
        Retrieve YouTube tokens for a user.
        """
        try:
            if not isinstance(user_id, str):
                user_id = str(user_id)

            collection = self._get_collection()
            token_doc = collection.find_one({"user_id": user_id})

            if not token_doc:
                return None

            creds = Credentials(
                token=token_doc.get("token"),
                refresh_token=token_doc.get("refresh_token"),
                token_uri=token_doc.get("token_uri"),
                client_id=token_doc.get("client_id"),
                client_secret=token_doc.get("client_secret"),
                scopes=token_doc.get("scopes"),
            )
            creds.expiry = token_doc.get("expiry")
            return creds

        except Exception as e:
            logger.error(f"Error retrieving YouTube tokens for user {user_id}: {str(e)}")
            return None

    async def refresh_tokens_if_needed(self, user_id: str) -> Optional[Credentials]:
        """
        Refreshes the user's youtube tokens if they are expired.
        """
        creds = await self.get_tokens(user_id)
        if not creds:
            return None

        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                await self.save_tokens(user_id, creds)
                logger.info(f"Refreshed and saved YouTube tokens for user {user_id}")
            except Exception as e:
                logger.error(f"Error refreshing YouTube tokens for user {user_id}: {e}")
                # If refresh fails, the creds object will not be valid.
                return None
        
        return creds

    async def delete_tokens(self, user_id: str) -> None:
        """
        Delete YouTube tokens for a user.
        """
        try:
            if not isinstance(user_id, str):
                user_id = str(user_id)

            collection = self._get_collection()
            result = collection.delete_one({"user_id": user_id})

            if result.deleted_count > 0:
                logger.info(f"YouTube tokens deleted for user {user_id}")
            else:
                logger.warning(f"No YouTube tokens found for user {user_id} to delete")

        except Exception as e:
            logger.error(f"Error deleting YouTube tokens for user {user_id}: {str(e)}")
            raise


youtube_token_service = YouTubeTokenService()
