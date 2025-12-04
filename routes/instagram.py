"""
Instagram OAuth2 integration routes.
Handles authentication, video posting, and status checking.
"""

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Annotated, List
from urllib.parse import urlencode, quote

import httpx
from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    UploadFile,
    File,
    Form,
    Depends,
    Query,
    status,
    Response,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from config import settings
from services.instagram_tokens import instagram_token_service
from services.auth import auth_service
from database.connection import get_database
from utils.s3_storage import s3_client
import os
import json
import uuid

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/instagram", tags=["instagram"])

# Security scheme
security = HTTPBearer()


# Pydantic models
class InstagramAuthUrlResponse(BaseModel):
    """Response model for Instagram auth URL"""

    auth_url: str
    state: str


class InstagramProfileResponse(BaseModel):
    """Response model for Instagram profile"""

    id: str
    username: str
    account_type: Optional[str] = None
    media_count: Optional[int] = None
    instagram_connected_at: Optional[datetime] = None


class InstagramCreatorInfoResponse(BaseModel):
    """Response model for creator info"""

    creator_username: Optional[str] = None
    creator_account_type: Optional[str] = None
    privacy_level_options: List[str] = []
    comment_disabled: bool = False
    max_video_post_duration_sec: int = 60


class InstagramPostResponse(BaseModel):
    """Response model for Instagram post"""

    publish_id: str
    status: str


class InstagramPostStatusResponse(BaseModel):
    """Response model for post status"""

    publish_id: str
    status: str
    progress: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# Helper functions
async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> Dict[str, Any]:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = auth_service.verify_token(credentials.credentials, "access")
    if token_data is None or token_data.user_id is None:
        raise credentials_exception

    user = await auth_service.get_user_by_id(token_data.user_id)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    normalized = auth_service.user_to_dict(user)
    try:
        return normalized if isinstance(normalized, dict) else normalized.model_dump()
    except Exception:
        return {
            "id": getattr(normalized, "id", None),
            "username": getattr(normalized, "username", None),
            "email": getattr(normalized, "email", None),
        }


def _extract_user_id(user_like: Any) -> Optional[str]:
    """Safely extract a user id from either a dict or pydantic model-like object."""
    try:
        if isinstance(user_like, dict):
            return (
                user_like.get("id") or user_like.get("_id") or user_like.get("user_id")
            )
        return (
            getattr(user_like, "id", None)
            or getattr(user_like, "_id", None)
            or getattr(user_like, "user_id", None)
        )
    except Exception:
        return None


async def _require_tokens(user_id: str) -> Dict[str, Any]:
    """Load Instagram tokens or raise 401"""
    tokens = await instagram_token_service.get_tokens(user_id)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Instagram account not connected. Please authenticate with Instagram first.",
        )
    return tokens


# Routes


@router.get("/auth-url")
async def get_instagram_auth_url(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate Instagram OAuth URL.
    """
    try:
        state = secrets.token_urlsafe(32)

        # Store state in DB
        db = get_database()
        db["oauth_states"].insert_one(
            {
                "state": state,
                "provider": "instagram",
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc).timestamp() + 600,
                "user_id": _extract_user_id(current_user),
            }
        )

        # Build Auth URL
        # https://api.instagram.com/oauth/authorize
        # ?client_id={app-id}&redirect_uri={redirect-uri}&scope=user_profile,user_media&response_type=code&state={state}

        params = {
            "client_id": settings.INSTA_APP_ID,
            "redirect_uri": settings.INSTA_REDIRECT_URI,
            # "scope": settings.INSTA_APP_SCOPE or "user_profile,user_media",
            "scope": "instagram_business_basic,instagram_business_content_publish",
            "response_type": "code",
            "state": state,
        }

        query_string = urlencode(params)
        auth_url = f"https://api.instagram.com/oauth/authorize?{query_string}"

        return {"auth_url": auth_url}

    except Exception as e:
        logger.error(f"Failed to generate Instagram auth URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate authorization URL",
        )


@router.get("/callback")
async def instagram_callback(
    code: Optional[str] = Query(None, description="Authorization code from Instagram"),
    state: Optional[str] = Query(
        None, description="State parameter for CSRF protection"
    ),
    error: Optional[str] = Query(None, description="Error from Instagram"),
    error_reason: Optional[str] = Query(None, description="Error reason"),
    error_description: Optional[str] = Query(None, description="Error description"),
):
    """
    Handle Instagram OAuth callback.
    """
    try:
        if error:
            logger.error(f"Instagram OAuth error: {error} - {error_description}")
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Instagram OAuth Error</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff4444;">❌ Instagram Authorization Failed</h1>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Description:</strong> {error_description or "No description provided"}</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)

        if not code or not state:
            return HTMLResponse("<h1>Missing parameters</h1>", status_code=400)

        # Validate state
        db = get_database()
        state_doc = db["oauth_states"].find_one(
            {"state": state, "provider": "instagram"}
        )

        if not state_doc:
            return HTMLResponse("<h1>Invalid or expired state</h1>", status_code=400)

        user_id = state_doc.get("user_id")
        if not user_id:
            return HTMLResponse("<h1>User context lost</h1>", status_code=400)

        # Exchange code for short-lived token
        async with httpx.AsyncClient() as client:
            # POST https://api.instagram.com/oauth/access_token
            data = {
                "client_id": settings.INSTA_APP_ID,
                "client_secret": settings.INSTA_APP_SECRET,
                "grant_type": "authorization_code",
                "redirect_uri": settings.INSTA_REDIRECT_URI,
                "code": code,
            }

            resp = await client.post(
                "https://api.instagram.com/oauth/access_token", data=data
            )
            if resp.status_code != 200:
                logger.error(f"Failed to exchange code: {resp.text}")
                return HTMLResponse(
                    f"<h1>Token exchange failed</h1><p>{resp.text}</p>", status_code=400
                )

            token_data = resp.json()
            short_lived_token = token_data.get("access_token")
            user_id_ig = token_data.get("user_id")  # Instagram User ID

            # Exchange for long-lived token
            # GET https://graph.instagram.com/access_token
            # ?grant_type=ig_exchange_token&client_secret={client-secret}&access_token={short-lived-token}

            long_lived_resp = await client.get(
                "https://api.instagram.com/oauth/access_token",
                params={
                    "grant_type": "ig_exchange_token",
                    "client_secret": settings.INSTA_APP_SECRET,
                    "access_token": short_lived_token,
                },
            )

            if long_lived_resp.status_code == 200:
                long_lived_data = long_lived_resp.json()
                access_token = long_lived_data.get("access_token")
                expires_in = long_lived_data.get("expires_in")  # seconds
                expires_at = datetime.now(timezone.utc).timestamp() + expires_in
            else:
                # Fallback to short lived if exchange fails (though unlikely if configured right)
                logger.warning(
                    f"Failed to exchange for long-lived token: {long_lived_resp.text}"
                )
                access_token = short_lived_token
                expires_at = (
                    datetime.now(timezone.utc).timestamp() + 3600
                )  # Assume 1 hour

            # Fetch User Profile
            # GET https://graph.instagram.com/me?fields=id,username,account_type,media_count&access_token={access-token}
            profile_resp = await client.get(
                "https://api.instagram.com/me",
                params={
                    "fields": "id,username,account_type,media_count",
                    "access_token": access_token,
                },
            )

            user_profile = {}
            if profile_resp.status_code == 200:
                user_profile = profile_resp.json()
            else:
                logger.error(f"Failed to fetch profile: {profile_resp.text}")

            # Save tokens
            await instagram_token_service.save_tokens(
                user_id,
                {
                    "access_token": access_token,
                    "user_profile": user_profile,
                    "expires_at": expires_at,
                },
            )

            # Clean up state
            db["oauth_states"].delete_one({"state": state})

            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Instagram Connected</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        text-align: center;
                        padding: 60px;
                        background: linear-gradient(45deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
                        color: white;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.2);
                        padding: 40px;
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                        display: inline-block;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Instagram Connected!</h2>
                    <p>Your account @{user_profile.get("username", "user")} is ready.</p>
                    <script>
                        if (window.opener) {{
                            window.opener.postMessage({{ type: 'INSTAGRAM_AUTH_SUCCESS' }}, '*');
                        }}
                        setTimeout(() => window.close(), 3000);
                    </script>
                </div>
            </body>
            </html>
            """)

    except Exception as e:
        logger.error(f"Instagram callback error: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


@router.get("/me", response_model=InstagramProfileResponse)
async def get_instagram_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get current user's Instagram connection status"""
    user_id = _extract_user_id(current_user)
    tokens = await instagram_token_service.get_tokens(user_id)

    if not tokens:
        raise HTTPException(status_code=404, detail="Instagram not connected")

    profile = tokens.get("user_profile", {})

    return InstagramProfileResponse(
        id=profile.get("id", ""),
        username=profile.get("username", ""),
        account_type=profile.get("account_type"),
        media_count=profile.get("media_count"),
        instagram_connected_at=tokens.get("created_at"),
    )


@router.post("/disconnect")
async def disconnect_instagram(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Disconnect Instagram account"""
    user_id = _extract_user_id(current_user)
    await instagram_token_service.delete_tokens(user_id)
    return {"message": "Disconnected successfully"}


# Content Posting Routes (Placeholder for now, will implement next)
@router.post("/creator-info", response_model=InstagramCreatorInfoResponse)
async def get_creator_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get posting capabilities"""
    user_id = _extract_user_id(current_user)
    tokens = await _require_tokens(user_id)
    profile = tokens.get("user_profile", {})

    return InstagramCreatorInfoResponse(
        creator_username=profile.get("username"),
        creator_account_type=profile.get("account_type"),
        privacy_level_options=[],  # Instagram API doesn't support privacy levels for API posts
        comment_disabled=False,
        max_video_post_duration_sec=60,  # Reels limit usually
    )


# @router.post("/post/video", response_model=InstagramPostResponse)
# async def post_video_to_instagram(
#     caption: str = Form(...),
#     video_url: str = Form(...),
#     cover_file: Optional[UploadFile] = File(None),
#     # To check status, we usually check the Container ID, but here we might only have Media ID.
#     # Actually, for Reels, the media_publish returns the Media ID.
#     # If we want to check status of the *upload*, we should probably check the Container ID.
#     # But the user flow usually returns the publish_id (Media ID) after publish.
#     # Let's assume we can query the Media ID for status.

#     async with httpx.AsyncClient() as client:
#         resp = await client.get(
#             f"https://graph.instagram.com/{publish_id}",
#             params={
#                 "fields": "status_code,status",  # These fields might be for Container, not Media.
#                 "access_token": access_token,
#             },
#         )

#         # If it's a Media ID, it might not have 'status_code'. It might just exist.
#         # If the ID is valid and returns data, it's likely published.

#         if resp.status_code == 200:
#             data = resp.json()
#             # If we get data back for a Media ID, it means it's published/created.
#             # Instagram Graph API doesn't have a granular "processing" status for Media ID
#             # the same way Container ID does.
#             # But if we want to be safe, we might need to store the Container ID.
#             # For now, let's assume if we can fetch it, it's FINISHED.

#             return InstagramPostStatusResponse(publish_id=publish_id, status="FINISHED")
#         else:
#             return InstagramPostStatusResponse(
#                 publish_id=publish_id, status="ERROR", error_message=resp.text
#             )
#         )
