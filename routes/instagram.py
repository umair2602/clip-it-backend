"""
Instagram OAuth2 integration routes.
Handles authentication, video posting, and status checking.
"""

import logging
import secrets
import time
import tempfile
import shutil
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
            "scope": settings.INSTA_APP_SCOPE or "public_profile",
            # "scope": ,
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
                "https://graph.instagram.com/access_token",
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
                "https://graph.instagram.com/me",
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


@router.get("/me")
async def get_instagram_me(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get current user's Instagram connection status"""
    user_id = _extract_user_id(current_user)
    tokens = await instagram_token_service.get_tokens(user_id)

    if not tokens:
        raise HTTPException(status_code=404, detail="Instagram not connected")

    # Debug: Return raw tokens to see what's going on
    return tokens


@router.post("/disconnect")
async def disconnect_instagram(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Disconnect Instagram account"""
    user_id = _extract_user_id(current_user)
    await instagram_token_service.delete_tokens(user_id)
    return {"message": "Disconnected successfully"}


# Content Posting Routes (Placeholder for now, will implement next)
import asyncio


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


@router.post("/post/video", response_model=InstagramPostResponse)
async def post_video_to_instagram(
    video_url: str = Form(...),
    caption: str = Form(...),
    title: str = Form(None),
    hashtags: str = Form(None),
    privacy_level: str = Form(None),
    disable_comments: bool = Form(False),
    schedule_date: str = Form(None),
    schedule_time: str = Form(None),
    cover_file: Optional[UploadFile] = File(None),
    video_cover_timestamp_ms: Optional[float] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Publishes a video to Instagram.
    Handles video URL, caption, hashtags, and optional cover image.
    """
    user_id = _extract_user_id(current_user)
    tokens = await _require_tokens(user_id)

    access_token = tokens.get("access_token")
    user_profile = tokens.get("user_profile", {})
    ig_user_id = user_profile.get("id")

    if not access_token or not ig_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Instagram credentials not found. Please reconnect.",
        )

    # 1. Prepare Caption
    full_caption = caption
    if title:
        full_caption = f"{title}\n\n{full_caption}"
    if hashtags:
        full_caption = f"{full_caption}\n\n{hashtags}"

    # 2. Handle Cover Image
    cover_url = None
    if cover_file:
        # Upload cover to S3
        file_ext = os.path.splitext(cover_file.filename)[1] or ".jpg"
        filename = f"ig_cover_{uuid.uuid4()}{file_ext}"

        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(cover_file.file, temp_file)
            temp_path = temp_file.name

        try:
            success, s3_key = s3_client.upload_file_to_s3(
                temp_path,
                user_id,
                filename,
                content_type=cover_file.content_type or "image/jpeg",
            )
            if success:
                # Get a presigned URL or public URL that Instagram can access
                # Instagram needs a public URL. If S3 is private, we need a presigned URL.
                cover_url = s3_client.get_object_url(
                    s3_key, presigned=True, expires_in=3600
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Create Media Container
    # POST https://graph.instagram.com/v21.0/{ig-user-id}/media
    container_url = f"https://graph.instagram.com/v21.0/{ig_user_id}/media"

    data = {
        "media_type": "REELS",  # Or VIDEO, but REELS is often preferred for short form
        "video_url": video_url,
        "caption": full_caption,
        "access_token": access_token,
    }

    # Add optional fields
    if cover_url:
        data["cover_url"] = cover_url
    elif video_cover_timestamp_ms is not None:
        data["thumb_offset"] = int(video_cover_timestamp_ms)

    # Note: privacy_level is not directly supported for standard API posting (it defaults to account setting)
    # disable_comments might be supported via 'comment_enabled' = !disable_comments if the API allows it for this endpoint.
    # Checking docs: 'comment_enabled' is not always available on creation for all endpoints, but we can try.
    # For now, we'll omit it to avoid errors unless we are sure.

    async with httpx.AsyncClient() as client:
        # Step A: Create Container
        logger.info(f"Creating IG media container for user {ig_user_id}")
        resp = await client.post(container_url, data=data)

        if resp.status_code != 200:
            logger.error(f"IG Create Container Failed: {resp.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create Instagram media container: {resp.text}",
            )

        container_id = resp.json().get("id")
        logger.info(f"Container created: {container_id}")

        # Return the container ID immediately so frontend can poll
        return InstagramPostResponse(
            publish_id=container_id,
            status="CREATED",
        )


@router.get("/container/{container_id}", response_model=InstagramPostStatusResponse)
async def get_container_status(
    container_id: str,
    access_token: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Check the status of an Instagram media container.
    """
    if not access_token:
        # Try to get from user tokens
        user_id = _extract_user_id(current_user)
        tokens = await instagram_token_service.get_tokens(user_id)
        if not tokens or not tokens.get("access_token"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Instagram not connected",
            )
        access_token = tokens["access_token"]

    status_url = f"https://graph.instagram.com/v21.0/{container_id}"
    params = {"fields": "status_code", "access_token": access_token}

    async with httpx.AsyncClient() as client:
        status_resp = await client.get(status_url, params=params)

        if status_resp.status_code != 200:
            logger.error(
                f"Container status check failed: {status_resp.status_code} - {status_resp.text}"
            )
            if status_resp.status_code == 403:
                # Check for rate limit error code
                error_data = status_resp.json()
                error_code = error_data.get("error", {}).get("code")
                if error_code == 4:
                    return InstagramPostStatusResponse(
                        publish_id=container_id,
                        status="RATE_LIMIT",
                        error_message="Instagram rate limit reached. Please wait.",
                    )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to check container status: {status_resp.text}",
            )

        status_json = status_resp.json()
        status_code_val = status_json.get("status_code")

        logger.info(f"Container {container_id} status: {status_code_val}")

        return InstagramPostStatusResponse(
            publish_id=container_id,
            status=status_code_val,  # IN_PROGRESS, FINISHED, ERROR
        )


@router.post("/publish/{container_id}", response_model=InstagramPostResponse)
async def publish_media(
    container_id: str,
    access_token: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """
    Publish an Instagram media container after it has finished processing.
    """
    if not access_token:
        # Try to get from user tokens
        user_id = _extract_user_id(current_user)
        tokens = await instagram_token_service.get_tokens(user_id)
        if not tokens or not tokens.get("access_token"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Instagram not connected",
            )
        access_token = tokens["access_token"]
        ig_user_id = tokens.get("user_profile", {}).get("id")
        if not ig_user_id:
            # Try to fetch profile if ID missing (shouldn't happen if auth flow worked)
            # Or we can decode token but that's complex.
            # Let's assume we have it or fetch it.
            pass  # For now assume we have it or can get it from /me

    # We need the user ID to publish. If it wasn't in tokens, we need to fetch it.
    # But wait, the publish endpoint is /{ig-user-id}/media_publish
    # So we definitely need it.

    if not access_token:  # Double check
        raise HTTPException(status_code=401, detail="No access token")

    async with httpx.AsyncClient() as client:
        # If we don't have ig_user_id, fetch it
        if not locals().get("ig_user_id"):
            me_resp = await client.get(
                f"https://graph.instagram.com/me?access_token={access_token}"
            )
            if me_resp.status_code == 200:
                ig_user_id = me_resp.json().get("id")
            else:
                raise HTTPException(
                    status_code=400, detail="Could not fetch Instagram User ID"
                )

        publish_url = f"https://graph.instagram.com/v21.0/{ig_user_id}/media_publish"
        publish_data = {"creation_id": container_id, "access_token": access_token}

        logger.info(f"Publishing IG media container {container_id}")
        # Publishing can take time, so we increase the timeout
        publish_resp = await client.post(publish_url, data=publish_data, timeout=60.0)

        if publish_resp.status_code != 200:
            logger.error(f"IG Publish Failed: {publish_resp.text}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to publish Instagram media: {publish_resp.text}",
            )

        publish_id = publish_resp.json().get("id")

        return InstagramPostResponse(
            publish_id=publish_id,
            status="PUBLISHED",
        )
