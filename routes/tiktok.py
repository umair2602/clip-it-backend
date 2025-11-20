"""
TikTok OAuth2 integration routes.
Handles authentication, video posting, and status checking.
"""

import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Annotated, List
from urllib.parse import urlencode, quote

import httpx
# from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, Form
from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Depends, Query, status, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from config import settings
from services.tiktok_tokens import tiktok_token_service
from services.auth import auth_service
import json


# Removed S3 import - keeping only auth functionality
from database.connection import get_database
import os

import base64
import hashlib

from io import BytesIO

# ? Proxy imports
from typing import Any, Dict, List
logger = logging.getLogger(__name__)
from starlette.responses import StreamingResponse


def generate_pkce_pair() -> tuple[str, str]:
    # Generate random code_verifier
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")
    # SHA256 -> base64url encoded
    code_challenge = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode("utf-8").rstrip("=")
    return code_verifier, code_challenge


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/tiktok", tags=["tiktok"])

@router.get("/debug-pkce/{state}")
async def debug_pkce(state: str):
    db = get_database()
    doc = db["oauth_states"].find_one({"state": state})
    return doc or {"error": "state not found"}


# Security scheme
security = HTTPBearer()


# TikTok verification file route
@router.get("/tiktok49Vbf7lAy5VdIKPhY9AR3kuO3sPGlwsX.txt")
async def serve_tiktok_verification():
    """Serve TikTok verification file for domain verification"""
    verification_content = (
        "tiktok-developers-site-verification=49Vbf7lAy5VdIKPhY9AR3kuO3sPGlwsX"
    )
    return Response(content=verification_content, media_type="text/plain")


# Alternative: Serve from static directory
@router.get("/static/tiktok49Vbf7lAy5VdIKPhY9AR3kuO3sPGlwsX.txt")
async def serve_tiktok_verification_static():
    """Serve TikTok verification file from static directory"""
    verification_path = "static/tiktok49Vbf7lAy5VdIKPhY9AR3kuO3sPGlwsX.txt"
    if os.path.exists(verification_path):
        return FileResponse(verification_path, media_type="text/plain")
    else:
        # Create the file if it doesn't exist
        os.makedirs("static", exist_ok=True)
        with open(verification_path, "w") as f:
            f.write(
                "tiktok-developers-site-verification=49Vbf7lAy5VdIKPhY9AR3kuO3sPGlwsX"
            )
        return FileResponse(verification_path, media_type="text/plain")


# TikTok verification file
VERIFICATION_FILENAME = "tiktok4QPEpS6YFmd4DmFfdr2Kjw4YKsWvEWky.txt"


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


# Pydantic models
class TikTokAuthUrlResponse(BaseModel):
    """Response model for TikTok auth URL"""

    auth_url: str
    state: str


class TikTokProfileResponse(BaseModel):
    """Response model for TikTok profile"""

    open_id: str
    display_name: str
    avatar_url: Optional[str] = None


# Removed creator info model - keeping only auth functionality


class TikTokPostRequest(BaseModel):
    """Request model for posting to TikTok"""

    title: str = Field(..., max_length=150, description="Video title")
    video_url: str = Field(..., description="S3 URL of the video to post")
    privacy_level: str = Field(
        "PUBLIC_TO_EVERYONE",
        description="Privacy level: PUBLIC_TO_EVERYONE, MUTUAL_FOLLOW_FRIENDS, SELF_ONLY",
    )
    disable_duet: bool = Field(False, description="Disable duet")
    disable_comment: bool = Field(False, description="Disable comments")
    disable_stitch: bool = Field(False, description="Disable stitch")
    video_cover_timestamp_ms: Optional[int] = Field(
        None, description="Video cover timestamp in milliseconds"
    )


class TikTokPostResponse(BaseModel):
    """Response model for TikTok post"""

    publish_id: str
    status: str


class TikTokPostStatusResponse(BaseModel):
    """Response model for post status"""

    publish_id: str
    status: str
    progress: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class TikTokCreatorInfoResponse(BaseModel):
    """Response model for creator info"""

    creator_avatar_url: Optional[str] = None
    creator_username: Optional[str] = None
    creator_nickname: Optional[str] = None
    privacy_level_options: List[str] = []
    comment_disabled: bool = False
    duet_disabled: bool = False
    stitch_disabled: bool = False
    max_video_post_duration_sec: Optional[int] = None


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

    # Always return a plain dict so downstream handlers can subscript
    # Normalize to plain dict for downstream usage
    normalized = auth_service.user_to_dict(user)
    try:
        return normalized if isinstance(normalized, dict) else normalized.model_dump()
    except Exception:
        # Fallback manual dict construction
        return {
            "id": getattr(normalized, "id", None),
            "username": getattr(normalized, "username", None),
            "first_name": getattr(normalized, "first_name", None),
            "last_name": getattr(normalized, "last_name", None),
            "email": getattr(normalized, "email", None),
            "is_active": getattr(normalized, "is_active", True),
            "privacy_accepted": getattr(normalized, "privacy_accepted", True),
            "videos": getattr(normalized, "videos", []),
            "created_at": getattr(normalized, "created_at", None),
            "updated_at": getattr(normalized, "updated_at", None),
        }


async def _require_tokens(user_id: str) -> Dict[str, Any]:
    """Load TikTok tokens or raise 401"""
    tokens = await tiktok_token_service.get_tokens(user_id)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="TikTok account not connected. Please authenticate with TikTok first.",
        )
    return tokens


# async def _refresh_if_needed(user_id: str) -> Dict[str, Any]:
#     """Refresh tokens if they're expired"""
#     tokens = await _require_tokens(user_id)

#     if await tiktok_token_service.is_token_expired(user_id):
#         logger.info(f"Refreshing expired tokens for user {user_id}")

#         try:
#             async with httpx.AsyncClient() as client:
#                 # Use new generation refresh token endpoint
#                 response = await client.post(
#                     f"{settings.TIKTOK_API_BASE}/oauth/refresh_token/",
#                     data={
#                         "client_key": settings.TIKTOK_CLIENT_KEY,
#                         "client_secret": settings.TIKTOK_CLIENT_SECRET,
#                         "grant_type": "refresh_token",
#                         "refresh_token": tokens["refresh_token"],
#                     },
#                 )

#                 if response.status_code != 200:
#                     logger.error(f"Failed to refresh TikTok tokens: {response.text}")
#                     raise HTTPException(
#                         status_code=status.HTTP_401_UNAUTHORIZED,
#                         detail="Failed to refresh TikTok tokens",
#                     )

#                 refresh_data = response.json()

#                 # Update tokens in database
#                 await tiktok_token_service.update_tokens(
#                     user_id,
#                     access_token=refresh_data["access_token"],
#                     refresh_token=refresh_data["refresh_token"],
#                     expires_at=int(time.time()) + refresh_data["expires_in"],
#                 )

#                 # Return updated tokens
#                 return await tiktok_token_service.get_tokens(user_id)

#         except Exception as e:
#             logger.error(f"Error refreshing TikTok tokens: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Failed to refresh TikTok tokens",
#             )

#     return tokens

async def _refresh_if_needed(user_id: str) -> Dict[str, Any]:
    """Refresh tokens if they're expired"""
    tokens = await _require_tokens(user_id)

    # --- Add logging to debug token info ---
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    open_id = tokens.get("open_id")
    scope = tokens.get("scope")
    expires_at = tokens.get("expires_at")
    
    logger.info(f"User {user_id} TikTok token info:")
    logger.info(f"  access_token: {str(access_token)[:10]}... (masked)")
    logger.info(f"  refresh_token: {str(refresh_token)[:10]}... (masked)")
    logger.info(f"  open_id: {open_id}")
    logger.info(f"  scope: {scope}")
    logger.info(f"  expires_at: {expires_at}")

    if await tiktok_token_service.is_token_expired(user_id):
        logger.info(f"Refreshing expired tokens for user {user_id}")

        try:
            async with httpx.AsyncClient() as client:
                # Use new generation refresh token endpoint
                response = await client.post(
                    f"{settings.TIKTOK_API_BASE}/oauth/refresh_token/",
                    data={
                        "client_key": settings.TIKTOK_CLIENT_KEY,
                        "client_secret": settings.TIKTOK_CLIENT_SECRET,
                        "grant_type": "refresh_token",
                        "refresh_token": tokens["refresh_token"],
                    },
                )

                if response.status_code != 200:
                    logger.error(f"Failed to refresh TikTok tokens: {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Failed to refresh TikTok tokens",
                    )

                refresh_data = response.json()

                # Update tokens in database
                await tiktok_token_service.update_tokens(
                    user_id,
                    access_token=refresh_data["access_token"],
                    refresh_token=refresh_data["refresh_token"],
                    expires_at=int(time.time()) + refresh_data["expires_in"],
                )

                # Log refreshed token info
                logger.info(f"Tokens refreshed for user {user_id}:")
                logger.info(f"  new access_token: {str(refresh_data['access_token'])[:10]}... (masked)")
                logger.info(f"  new refresh_token: {str(refresh_data['refresh_token'])[:10]}... (masked)")
                logger.info(f"  expires_in: {refresh_data['expires_in']} seconds")

                # Return updated tokens
                return await tiktok_token_service.get_tokens(user_id)

        except Exception as e:
            logger.error(f"Error refreshing TikTok tokens: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to refresh TikTok tokens",
            )

    return tokens


async def _validate_video_url(video_url: str) -> str:
    """Validate and return the video URL for TikTok posting"""
    if not video_url:
        raise HTTPException(status_code=400, detail="Video URL is required")

    if not video_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Video URL must be HTTPS")

    # Ensure it's from a verified domain (S3 in this case)
    if "s3.amazonaws.com" not in video_url and "s3." not in video_url:
        logger.warning(f"Video URL may not be from verified domain: {video_url}")

    return video_url


async def _preflight_check_video_url(video_url: str) -> None:
    """Perform a lightweight HEAD/GET to ensure TikTok can fetch the video URL.

    Validates:
      - URL responds with 200-OK on HEAD (or supports range GET)
      - Content-Type looks like a video or octet-stream
      - Content-Length is present (when provided by server)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            # Prefer HEAD
            head_resp = await client.head(video_url)
            if head_resp.status_code in (200, 204):
                content_type = head_resp.headers.get("Content-Type", "").lower()
                content_length = head_resp.headers.get("Content-Length")
            else:
                # Some CDNs/S3 presigned URLs don't allow HEAD; try range GET
                get_resp = await client.get(video_url, headers={"Range": "bytes=0-0"})
                if get_resp.status_code not in (200, 206):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Video URL is not publicly accessible",
                            "hint": "Ensure the S3 object is accessible via the provided HTTPS URL or use FILE_UPLOAD mode.",
                            "status": get_resp.status_code,
                        },
                    )
                content_type = get_resp.headers.get("Content-Type", "").lower()
                content_length = get_resp.headers.get("Content-Length")

            if not (
                content_type.startswith("video/")
                or content_type.startswith("application/octet-stream")
            ):
                # Allow empty content-type but warn
                if content_type:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Video URL does not return a video content type",
                            "content_type": content_type,
                            "hint": "Set correct Content-Type for the object or switch to FILE_UPLOAD mode.",
                        },
                    )

            # Optionally ensure non-zero size when provided
            if content_length is not None:
                try:
                    if int(content_length) <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "message": "Video URL has zero content length",
                                "hint": "Check S3 object and permissions.",
                            },
                        )
                except ValueError:
                    # Ignore unparsable Content-Length
                    pass
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Preflight check failed for video URL {video_url}: {e}")
        # Do not block hard on network quirks; provide a helpful error to the client
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Failed to validate video URL reachability",
                "hint": "Ensure the URL is publicly reachable by TikTok (domain verification may be required for PULL_FROM_URL).",
            },
        )


# TikTok verification endpoint
@router.get(f"/callback/{VERIFICATION_FILENAME}", include_in_schema=False)
def serve_tiktok_verification():
    """Serve TikTok verification file at callback path"""
    from pathlib import Path

    verification_path = Path(__file__).parent.parent / "verify" / VERIFICATION_FILENAME
    if not verification_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Verification file not found at {verification_path}",
        )
    return FileResponse(verification_path, media_type="text/plain")


# Routes
@router.get("/debug-config")
async def debug_tiktok_config():
    """Debug endpoint to show current TikTok configuration (remove in production)"""
    return {
        "TIKTOK_CLIENT_KEY": settings.TIKTOK_CLIENT_KEY,
        "TIKTOK_REDIRECT_URI": settings.TIKTOK_REDIRECT_URI,
        "TIKTOK_AUTH_BASE": settings.TIKTOK_AUTH_BASE,
        "TIKTOK_API_BASE": settings.TIKTOK_API_BASE,
        "TIKTOK_SCOPES": settings.TIKTOK_SCOPES,
    }


@router.get("/auth-url")
async def get_tiktok_auth_url(
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_verifier: Optional[str] = Query(None, description="PKCE code verifier"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate TikTok OAuth URL and return access token after Login Kit flow.
    """
    try:
        state = secrets.token_urlsafe(32)

        # Generate PKCE pair if not provided
        if not code_challenge or not code_verifier:
            code_verifier, code_challenge = generate_pkce_pair()

        # Store state + PKCE in DB
        db = get_database()
        db["oauth_states"].insert_one({
            "state": state,
            "code_challenge": code_challenge,
            "code_verifier": code_verifier,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc).timestamp() + 600,
            "user_id": _extract_user_id(current_user),
        })

        # Build Login Kit URL
        scopes_value = settings.TIKTOK_SCOPES
        # scopes_value = "user.info.basic, video.publish"
        params = {
            "client_key": settings.TIKTOK_CLIENT_KEY,
            "scope": scopes_value,
            "response_type": "code",
            "redirect_uri": settings.TIKTOK_REDIRECT_URI,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        query_string = "&".join(
            f"{k}={quote(str(v), safe='')}" for k, v in params.items() if v is not None
        )
        auth_url = f"{settings.TIKTOK_AUTH_BASE}/auth/authorize/?{query_string}"

        # Instead of logs, return the auth URL and current stored token (if exists)
        tokens = await tiktok_token_service.get_tokens(_extract_user_id(current_user))
        access_token = tokens.get("access_token") if tokens else None

        return {
            "auth_url": auth_url,
            "access_token": access_token,
            "state": state,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate authorization URL",
        )


@router.get("/callback/")
@router.get("/callback")
async def tiktok_callback(
    code: Optional[str] = Query(None, description="Authorization code from TikTok"),
    state: Optional[str] = Query(
        None, description="State parameter for CSRF protection"
    ),
    error: Optional[str] = Query(None, description="Error from TikTok"),
    error_description: Optional[str] = Query(
        None, description="Error description from TikTok"
    ),
):
    """
    Handle TikTok OAuth callback according to Login Kit for Web documentation.
    Validates state, exchanges code for tokens, and provides user feedback.
    """
    try:
        # Check for OAuth errors first
        if error:
            logger.error(f"TikTok OAuth error: {error} - {error_description}")
            return HTMLResponse(f"""
            <!DOCTYPE html>
            <html>
            <head><title>TikTok OAuth Error</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff4444;">❌ TikTok Authorization Failed</h1>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Description:</strong> {error_description or "No description provided"}</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)

        # Validate required parameters
        if not code or not state:
            logger.error("Missing required OAuth parameters: code or state")
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Missing Parameters</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff8800;">⚠️ Missing OAuth Parameters</h1>
                <p>Authorization code or state parameter is missing.</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)

        # Validate state parameter (CSRF protection)
        db = get_database()
        state_doc = db["oauth_states"].find_one({"state": state})

        if not state_doc:
            logger.error(f"Invalid state parameter: {state}")
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Invalid State</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff8800;">⚠️ Invalid State Parameter</h1>
                <p>The OAuth state parameter is invalid or expired.</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)

        # Check if state is expired
        if state_doc["expires_at"] < datetime.now(timezone.utc).timestamp():
            logger.error(f"State parameter expired: {state}")
            # Clean up expired state
            db["oauth_states"].delete_one({"state": state})
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head><title>Expired State</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1 style="color: #ff8800;">⚠️ OAuth State Expired</h1>
                <p>The OAuth state parameter has expired. Please try again.</p>
                <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                <script>setTimeout(() => window.close(), 5000);</script>
            </body>
            </html>
            """)

        # Do not delete the state yet; it's used to bind user_id in token save

        # Retrieve stored PKCE code_verifier from state document
        stored_code_verifier = state_doc.get("code_verifier")

        # Prepare token exchange data according to TikTok API docs
        token_exchange_data = {
            "client_key": settings.TIKTOK_CLIENT_KEY,
            "client_secret": settings.TIKTOK_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.TIKTOK_REDIRECT_URI,
        }

        # Add PKCE code_verifier if it was stored with the state
        if stored_code_verifier:
            token_exchange_data["code_verifier"] = stored_code_verifier
            logger.info("Using stored PKCE code_verifier for token exchange")
        else:
            logger.error("No PKCE code_verifier found in stored state")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PKCE code_verifier not found in state",
            )

        logger.info(f"Exchanging authorization code for tokens...")

        # Exchange code for tokens using TikTok API v2
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.TIKTOK_API_BASE}/oauth/token/",
                data=token_exchange_data,
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"Failed to exchange code for tokens: {response.status_code} - {response.text}"
                )
                return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head><title>Token Exchange Failed</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #ff4444;">❌ Token Exchange Failed</h1>
                    <p>Failed to exchange authorization code for tokens.</p>
                    <p><strong>Status:</strong> {response.status_code}</p>
                    <p><strong>Error:</strong> {response.text[:200]}</p>
                    <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                    <script>setTimeout(() => window.close(), 5000);</script>
                </body>
                </html>
                """)

            token_data = response.json()
            logger.info("Successfully exchanged code for tokens")

            # ⚠️ For testing only — log the access token
            logger.info(f"TikTok Access Token (TEST ONLY): {token_data.get('access_token')}")
            print("TikTok Access Token (TEST ONLY):", token_data.get('access_token'))

            # Validate token response contains required fields
            if "access_token" not in token_data:
                logger.error(
                    f"Token exchange response missing access_token: {token_data}"
                )
                return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head><title>Token Exchange Failed</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #ff4444;">❌ Token Exchange Failed</h1>
                    <p>Token response is missing access_token.</p>
                    <p><strong>Response:</strong> {str(token_data)[:200]}</p>
                    <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                    <script>setTimeout(() => window.close(), 5000);</script>
                </body>
                </html>
                """)

            # Extract user info from access token (new API requires fields)
            user_info_response = await client.get(
                f"{settings.TIKTOK_API_BASE}/user/info/",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
                params={"fields": "open_id,display_name,avatar_url"},
                timeout=30,
            )

            if user_info_response.status_code != 200:
                logger.error(
                    f"Failed to get user info: {user_info_response.status_code} - {user_info_response.text}"
                )
                return HTMLResponse("""
                <!DOCTYPE html>
                <html>
                <head><title>User Info Failed</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1 style="color: #ff4444;">❌ Failed to Get User Information</h1>
                    <p>Could not retrieve user information from TikTok.</p>
                    <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
                    <script>setTimeout(() => window.close(), 5000);</script>
                </body>
                </html>
                """)

            user_info = user_info_response.json()
            user_data = user_info.get("data", {}).get("user", {})
            open_id = user_data.get("open_id")
            if not open_id:
                logger.error(
                    f"Missing open_id in TikTok user info response: {user_info}"
                )
                raise HTTPException(
                    status_code=400, detail="TikTok user info missing open_id"
                )
            logger.info(f"Successfully retrieved user info for open_id: {open_id}")

            # Resolve the user that initiated the auth by matching the state record
            db = get_database()
            state_doc = db["oauth_states"].find_one({"state": state})
            user_id = state_doc.get("user_id") if state_doc else None
            if not user_id:
                # Fallback to deny if we can't associate to a user
                logger.error("Missing user_id for TikTok callback; state not bound")
                raise HTTPException(
                    status_code=400, detail="Invalid auth state: missing user context"
                )

            # Compute expires_at = now + expires_in
            expires_at = int(time.time()) + token_data["expires_in"]

            # Save tokens to database
            await tiktok_token_service.save_tokens(
                user_id,
                {
                    "open_id": open_id,
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data["refresh_token"],
                    "scope": token_data["scope"],
                    "expires_at": expires_at,
                },
            )

            # Clean up state after successful binding and save
            try:
                db["oauth_states"].delete_one({"state": state})
            except Exception:
                pass

            logger.info(
                f"TikTok OAuth successful for user {user_id}, open_id: {open_id}"
            )

            # Return success page that can close itself and communicate with parent
            return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>TikTok Connected</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            text-align: center;
            padding: 60px;
            background: linear-gradient(135deg, #9581e6 0%, #6e5bb3 100%);
            color: white;
        }}
        .container {{
            background: rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            display: inline-block;
            min-width: 300px;
        }}
        h1 {{
            font-size: 3em;
            margin-bottom: 20px;
        }}
        h2 {{
            margin-bottom: 15px;
            font-weight: normal;
        }}
        p {{
            font-size: 16px;
            margin-bottom: 20px;
        }}
        .info-box {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            word-break: break-all;
        }}
        .small-text {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>TikTok Connected!</h2>
        <p>Your account is ready to use.</p>
        <div class="info-box">
            <p><strong>Open ID:</strong> {open_id}</p>
        </div>
        <p class="small-text">This window will close automatically in 3 seconds.</p>
    </div>
    <script>
        // Notify parent window of successful connection
        if (window.opener) {{
            try {{
                window.opener.postMessage({{
                    type: 'TIKTOK_AUTH_SUCCESS',
                    data: {{
                        open_id: '{open_id}'
                    }}
                }}, '*');
            }} catch (e) {{
                console.log('Could not post message to parent:', e);
            }}
        }}
        
        // Close the popup window after 3 seconds
        setTimeout(() => {{
            if (window.opener) {{
                window.close();
            }} else {{
                window.location.href = 'https://clip-it-frontend-nine.vercel.app/history';
            }}
        }}, 3000);
    </script>
</body>
</html>
""")


    except Exception as e:
        logger.error(f"Error in TikTok callback: {str(e)}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: #ff4444;">❌ OAuth Error</h1>
            <p>An error occurred during TikTok authorization.</p>
            <p><strong>Error:</strong> {str(e)}</p>
            <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
            <script>setTimeout(() => window.close(), 5000);</script>
        </body>
        </html>
        """)


@router.get("/me", response_model=TikTokProfileResponse)
async def get_tiktok_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get TikTok profile information for the authenticated user.

    Returns:
        TikTok profile data (open_id, display_name, avatar_url)
    """
    try:
        # Ensure dict shape
        if not isinstance(current_user, dict):
            try:
                current_user = current_user.model_dump()
            except Exception:
                current_user = {
                    "id": getattr(current_user, "id", None),
                    "username": getattr(current_user, "username", None),
                }
        user_id = _extract_user_id(current_user)
        if not user_id:
            logger.error("Could not extract user_id from current_user for /tiktok/me")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context"
            )
        tokens = await _refresh_if_needed(user_id)

        if tokens:
            logger.info(f"TikTok access token for user {_extract_user_id(current_user)}: {tokens}")


        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.TIKTOK_API_BASE}/user/info/",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                params={"fields": "open_id,display_name,avatar_url"},
            )

            if response.status_code != 200:
                logger.error(f"Failed to get TikTok profile: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get TikTok profile",
                )

            user_info = response.json()
            data = user_info.get("data", {}).get("user", {})

            return TikTokProfileResponse(
                open_id=data.get("open_id"),
                display_name=data.get("display_name"),
                avatar_url=data.get("avatar_url"),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting TikTok profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/creator-info", response_model=TikTokCreatorInfoResponse)
async def get_creator_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get TikTok creator information and privacy options"""
    try:
        user_id = _extract_user_id(current_user)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user context")

        tokens = await _refresh_if_needed(user_id)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.TIKTOK_API_BASE}/post/publish/creator_info/query/",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
                json={},
            )

            if response.status_code != 200:
                logger.error(
                    f"Failed to get creator info: {response.status_code} - {response.text}"
                )
                raise HTTPException(
                    status_code=400, detail="Failed to get creator information"
                )

            creator_info = response.json()
            data = creator_info.get("data", {})

            return TikTokCreatorInfoResponse(
                creator_avatar_url=data.get("creator_avatar_url"),
                creator_username=data.get("creator_username"),
                creator_nickname=data.get("creator_nickname"),
                privacy_level_options=data.get("privacy_level_options", []),
                comment_disabled=data.get("comment_disabled", False),
                duet_disabled=data.get("duet_disabled", False),
                stitch_disabled=data.get("stitch_disabled", False),
                max_video_post_duration_sec=data.get("max_video_post_duration_sec"),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting creator info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# @router.post("/post/video", response_model=TikTokPostResponse)
# async def post_video_to_tiktok(
#     post_request: TikTokPostRequest,
#     current_user: Dict[str, Any] = Depends(get_current_user),
# ):
#     """Post a video to TikTok using PULL_FROM_URL from S3"""

#     # --- ADD THIS ---
#     user_id = _extract_user_id(current_user)
#     logger.info(f"POST /post/video called by user_id: {user_id}")
#     logger.info(f"Post request payload: {post_request.dict() if hasattr(post_request, 'dict') else post_request}")

#     try:
#         user_id = _extract_user_id(current_user)
#         if not user_id:
#             raise HTTPException(status_code=401, detail="Invalid user context")

#         tokens = await _refresh_if_needed(user_id)
#         logger.info(f"Using TikTok access token: {tokens['access_token']}")

#         # Validate video URL
#         video_url = await _validate_video_url(post_request.video_url)
#         await _preflight_check_video_url(video_url)

#         # Fetch creator info to validate privacy level and constraints
#         allowed_privacy_levels: List[str] = []
#         creator_constraints: Dict[str, Any] = {}
#         try:
#             async with httpx.AsyncClient() as client:
#                 ci_resp = await client.post(
#                     f"{settings.TIKTOK_API_BASE}/post/publish/creator_info/query/",
#                     headers={
#                         "Authorization": f"Bearer {tokens['access_token']}",
#                         "Content-Type": "application/json; charset=UTF-8",
#                     },
#                     json={},
#                 )
#                 if ci_resp.status_code == 200:
#                     ci_json = ci_resp.json()
#                     data = ci_json.get("data", {})
#                     allowed_privacy_levels = data.get("privacy_level_options", []) or []
#                     creator_constraints = {
#                         "comment_disabled": data.get("comment_disabled", False),
#                         "duet_disabled": data.get("duet_disabled", False),
#                         "stitch_disabled": data.get("stitch_disabled", False),
#                         "max_video_post_duration_sec": data.get(
#                             "max_video_post_duration_sec"
#                         ),
#                     }
#                 else:
#                     logger.warning(
#                         f"creator_info query failed ({ci_resp.status_code}): {ci_resp.text}"
#                     )
#         except Exception as e:
#             logger.warning(f"creator_info query error: {e}")

#         # If creator_info provided options, enforce them to avoid TikTok 403
#         if allowed_privacy_levels:
#             if post_request.privacy_level not in allowed_privacy_levels:
#                 logger.error(
#                     f"Requested privacy_level '{post_request.privacy_level}' not in allowed options: {allowed_privacy_levels}"
#                 )
#                 raise HTTPException(
#                     status_code=400,
#                     detail={
#                         "message": "Invalid privacy_level for this creator",
#                         "allowed_privacy_levels": allowed_privacy_levels,
#                         "hint": "For unaudited clients you must use SELF_ONLY and the account must be set to private.",
#                     },
#                 )

#         # Apply interaction constraints if provided (defensive)
#         if creator_constraints:
#             if creator_constraints.get("comment_disabled"):
#                 post_request.disable_comment = True
#             if creator_constraints.get("duet_disabled"):
#                 post_request.disable_duet = True
#             if creator_constraints.get("stitch_disabled"):
#                 post_request.disable_stitch = True

#         # Prepare post data for PULL_FROM_URL
#         post_data = {
#             "post_info": {
#                 "title": post_request.title,
#                 "privacy_level": post_request.privacy_level,
#                 "disable_duet": post_request.disable_duet,
#                 "disable_comment": post_request.disable_comment,
#                 "disable_stitch": post_request.disable_stitch,
#             },
#             "source_info": {"source": "PULL_FROM_URL", "video_url": video_url},
#         }

#         logger.info(f"Posting to TikTok with data: {post_data}")

#         # Add video cover timestamp if provided
#         if post_request.video_cover_timestamp_ms:
#             post_data["post_info"]["video_cover_timestamp_ms"] = (
#                 post_request.video_cover_timestamp_ms
#             )

#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 f"{settings.TIKTOK_API_BASE}/post/publish/video/init/",
#                 headers={
#                     "Authorization": f"Bearer {tokens['access_token']}",
#                     "Content-Type": "application/json; charset=UTF-8",
#                 },
#                 json=post_data,
#             )

#             logger.info(f"TikTok response status: {response.status_code}")
#             logger.info(f"TikTok response body: {response.text}")


#             if response.status_code != 200:
#                 # Try to parse error details
#                 error_payload = None
#                 try:
#                     error_payload = response.json()
#                 except Exception:
#                     error_payload = {"raw": response.text}
#                 error_obj = (
#                     (error_payload or {}).get("error", {})
#                     if isinstance(error_payload, dict)
#                     else {}
#                 )
#                 log_id = (
#                     error_obj.get("log_id")
#                     or (error_payload or {}).get("log_id")
#                     or "unknown"
#                 )
#                 code = error_obj.get("code") or (error_payload or {}).get("code")
#                 message = (
#                     error_obj.get("message")
#                     or (error_payload or {}).get("message")
#                     or "Failed to initialize TikTok post"
#                 )
#                 logger.error(
#                     f"Failed to initialize TikTok post (status={response.status_code}, log_id={log_id}, code={code}): {response.text}"
#                 )
#                 # Provide actionable hint for common unaudited-client error
#                 hint = None
#                 if code == "unaudited_client_can_only_post_to_private_accounts":
#                     hint = "Set the TikTok account to Private and use privacy_level=SELF_ONLY (unaudited clients)."
#                 raise HTTPException(
#                     status_code=400,
#                     detail={
#                         "message": message,
#                         "code": code,
#                         "log_id": log_id,
#                         "status": response.status_code,
#                         **({"hint": hint} if hint else {}),
#                     },
#                 )

#             post_result = response.json()
#             data = post_result.get("data", {})

#             return TikTokPostResponse(
#                 publish_id=data.get("publish_id"), status="initialized"
#             )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error posting to TikTok: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/post/video", response_model=TikTokPostResponse)
async def post_video_to_tiktok(
    post_request: TikTokPostRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Post a video to TikTok using PULL_FROM_URL from S3"""

    user_id = _extract_user_id(current_user)
    logger.info(f"POST /post/video called by user_id: {user_id}")
    logger.info(
        f"Post request payload: {post_request.dict() if hasattr(post_request, 'dict') else post_request}"
    )

    try:
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user context")

        tokens = await _refresh_if_needed(user_id)
        logger.info(f"Using TikTok access token: {tokens['access_token']}")

        # --- ensure we have a proxy URL for S3-hosted videos ---
        original_video_url = post_request.video_url
        if original_video_url:
            proxied = build_proxy_url_from_s3(original_video_url)
            if proxied != original_video_url:
                logger.info(
                    f"Rewriting video URL for TikTok pull_from_url: {original_video_url} -> {proxied}"
                )
                post_request.video_url = proxied

        # Validate video URL
        video_url = await _validate_video_url(post_request.video_url)
        await _preflight_check_video_url(video_url)

        # Fetch creator info to validate privacy level and constraints
        allowed_privacy_levels: List[str] = []
        creator_constraints: Dict[str, Any] = {}
        try:
            async with httpx.AsyncClient() as client:
                ci_resp = await client.post(
                    f"{settings.TIKTOK_API_BASE}/post/publish/creator_info/query/",
                    headers={
                        "Authorization": f"Bearer {tokens['access_token']}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json={},
                )
                if ci_resp.status_code == 200:
                    ci_json = ci_resp.json()
                    data = ci_json.get("data", {})
                    allowed_privacy_levels = data.get("privacy_level_options", []) or []
                    creator_constraints = {
                        "comment_disabled": data.get("comment_disabled", False),
                        "duet_disabled": data.get("duet_disabled", False),
                        "stitch_disabled": data.get("stitch_disabled", False),
                        "max_video_post_duration_sec": data.get(
                            "max_video_post_duration_sec"
                        ),
                    }
                else:
                    logger.warning(
                        f"creator_info query failed ({ci_resp.status_code}): {ci_resp.text}"
                    )
        except Exception as e:
            logger.warning(f"creator_info query error: {e}")

        # Enforce privacy levels and interaction constraints
        if allowed_privacy_levels:
            if post_request.privacy_level not in allowed_privacy_levels:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Invalid privacy_level for this creator",
                        "allowed_privacy_levels": allowed_privacy_levels,
                        "hint": "For unaudited clients you must use SELF_ONLY and the account must be set to private.",
                    },
                )

        if creator_constraints:
            if creator_constraints.get("comment_disabled"):
                post_request.disable_comment = True
            if creator_constraints.get("duet_disabled"):
                post_request.disable_duet = True
            if creator_constraints.get("stitch_disabled"):
                post_request.disable_stitch = True

        # Prepare post data for TikTok PULL_FROM_URL
        post_data = {
            "post_info": {
                "title": post_request.title,
                "privacy_level": post_request.privacy_level,
                "disable_duet": post_request.disable_duet,
                "disable_comment": post_request.disable_comment,
                "disable_stitch": post_request.disable_stitch,
            },
            "source_info": {"source": "PULL_FROM_URL", "video_url": video_url},
        }

        if post_request.video_cover_timestamp_ms:
            post_data["post_info"]["video_cover_timestamp_ms"] = (
                post_request.video_cover_timestamp_ms
            )

        logger.info(f"Posting to TikTok with data: {post_data}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.TIKTOK_API_BASE}/post/publish/video/init/",
                headers={
                    "Authorization": f"Bearer {tokens['access_token']}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json=post_data,
            )

            logger.info(f"TikTok response status: {response.status_code}")
            logger.info(f"TikTok response body: {response.text}")

            if response.status_code != 200:
                try:
                    error_payload = response.json()
                except Exception:
                    error_payload = {"raw": response.text}

                error_obj = (
                    (error_payload or {}).get("error", {})
                    if isinstance(error_payload, dict)
                    else {}
                )
                log_id = (
                    error_obj.get("log_id")
                    or (error_payload or {}).get("log_id")
                    or "unknown"
                )
                code = error_obj.get("code") or (error_payload or {}).get("code")
                message = (
                    error_obj.get("message")
                    or (error_payload or {}).get("message")
                    or "Failed to initialize TikTok post"
                )

                hint = None
                if code == "unaudited_client_can_only_post_to_private_accounts":
                    hint = "Set the TikTok account to Private and use privacy_level=SELF_ONLY (unaudited clients)."

                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": message,
                        "code": code,
                        "log_id": log_id,
                        "status": response.status_code,
                        **({"hint": hint} if hint else {}),
                    },
                )

            post_result = response.json()
            data = post_result.get("data", {})

            return TikTokPostResponse(
                publish_id=data.get("publish_id"), status="initialized"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error posting to TikTok: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/post/status/{publish_id}")
async def get_post_status(
    publish_id: str, current_user: Annotated[Dict[str, Any], Depends(get_current_user)]
) -> TikTokPostStatusResponse:
    """Get the status of a TikTok post"""
    try:
        logger.info(
            f"Getting post status for {publish_id} for user {current_user.get('id')}"
        )

        # Validate publish_id
        if not publish_id or publish_id.strip() == "":
            raise HTTPException(status_code=400, detail="publish_id is required")

        # Get user's TikTok tokens
        user_id = current_user.get("id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid user ID")

        tokens = await _require_tokens(user_id)
        logger.info(f"Retrieved tokens for user {user_id}")

        # Call TikTok API to get status
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.TIKTOK_API_BASE}/post/publish/status/fetch/",
                    headers={
                        "Authorization": f"Bearer {tokens['access_token']}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json={"publish_id": publish_id},
                )

                logger.info(f"TikTok API response status: {response.status_code}")

                if response.status_code != 200:
                    response_data = response.json()
                    log_id = response_data.get("error", {}).get("log_id", "unknown")
                    error_code = response_data.get("error", {}).get("code", "unknown")
                    error_message = response_data.get("error", {}).get(
                        "message", "Unknown error"
                    )

                    logger.error(
                        f"TikTok API error: {response.status_code} - {error_message} (log_id: {log_id})"
                    )

                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": f"TikTok API error: {error_message}",
                            "code": error_code,
                            "log_id": log_id,
                            "status": response.status_code,
                            "hint": "Check TikTok API response for details",
                        },
                    )

                # Parse TikTok response
                try:
                    status_result = response.json()
                    logger.info(f"TikTok status response: {status_result}")
                except Exception as e:
                    logger.error(f"Failed to parse TikTok response as JSON: {e}")
                    logger.error(f"Response text: {response.text}")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": "Failed to parse TikTok response",
                            "hint": "TikTok returned invalid JSON",
                        },
                    )

                data = status_result.get("data", {})

                # Debug log the TikTok response
                logger.info(f"TikTok status response for {publish_id}: {status_result}")

                # TikTok status response may not include publish_id; use path param
                try:
                    # Ensure publish_id is always a string
                    safe_publish_id = str(publish_id) if publish_id else "unknown"

                    # Validate required fields first
                    status_value = data.get("status")
                    if not status_value:
                        logger.warning(
                            f"TikTok response missing status field, using UNKNOWN"
                        )
                        status_value = "UNKNOWN"

                    # Debug logging after status_value is defined
                    logger.info(
                        f"DEBUG: publish_id param = {publish_id} (type: {type(publish_id)})"
                    )
                    logger.info(
                        f"DEBUG: safe_publish_id = {safe_publish_id} (type: {type(safe_publish_id)})"
                    )
                    logger.info(
                        f"DEBUG: status_value = {status_value} (type: {type(status_value)})"
                    )
                    logger.info(
                        f"DEBUG: progress = {data.get('progress')} (type: {type(data.get('progress'))})"
                    )
                    logger.info(
                        f"DEBUG: downloaded_bytes = {data.get('downloaded_bytes')} (type: {type(data.get('downloaded_bytes'))})"
                    )

                    # Test model creation with simple values
                    try:
                        test_obj = TikTokPostStatusResponse(
                            publish_id="test_id", status="test_status"
                        )
                        logger.info(
                            f"DEBUG: Test model creation successful: {test_obj}"
                        )
                    except Exception as e:
                        logger.error(f"DEBUG: Test model creation failed: {e}")

                    # Create response object
                    response_obj = TikTokPostStatusResponse(
                        publish_id=safe_publish_id,
                        status=status_value,
                        progress=data.get("progress") or data.get("downloaded_bytes"),
                        error_code=data.get("error_code"),
                        error_message=data.get("error_message"),
                    )

                    logger.info(f"Successfully created status response: {response_obj}")
                    return response_obj

                except Exception as e:
                    logger.error(f"Pydantic validation error for status response: {e}")
                    logger.error(
                        f"Data being validated: publish_id={publish_id}, status={data.get('status')}, progress={data.get('progress')}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "message": f"Failed to parse TikTok status response: {str(e)}",
                            "hint": "Check TikTok API response format",
                            "publish_id": str(publish_id) if publish_id else "unknown",
                        },
                    )

        except httpx.RequestError as e:
            logger.error(f"HTTP request error to TikTok API: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to connect to TikTok API",
                    "hint": "Network or connection issue",
                },
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_post_status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Internal server error: {str(e)}",
                "hint": "Check server logs for details",
            },
        )


@router.post("/disconnect")
async def disconnect_tiktok(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Disconnect TikTok account and revoke tokens.
    """
    try:
        if not isinstance(current_user, dict):
            try:
                current_user = current_user.model_dump()
            except Exception:
                current_user = {
                    "id": getattr(current_user, "id", None),
                    "username": getattr(current_user, "username", None),
                }
        user_id = _extract_user_id(current_user)
        if not user_id:
            logger.error(
                "Could not extract user_id from current_user for /tiktok/disconnect"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context"
            )
        tokens = await tiktok_token_service.get_tokens(user_id)

        if tokens:
            # Try to revoke tokens on TikTok side using new generation API
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{settings.TIKTOK_API_BASE}/oauth/revoke/",
                        data={
                            "client_key": settings.TIKTOK_CLIENT_KEY,
                            "client_secret": settings.TIKTOK_CLIENT_SECRET,
                            "token": tokens["access_token"],
                        },
                    )

                if response.status_code != 200:
                    logger.warning(f"Failed to revoke TikTok tokens: {response.text}")
            except Exception as e:
                logger.warning(f"Error revoking TikTok tokens: {str(e)}")

            # Delete tokens from our database
            await tiktok_token_service.delete_tokens(user_id)

        return {"message": "TikTok account disconnected successfully"}

    except Exception as e:
        logger.error(f"Error disconnecting TikTok: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

@router.post("/share")
async def share_to_tiktok(request: Request):
    """
    Share a video to TikTok using the access token from sandbox.
    The user must have connected their TikTok account before.
    """
    data = await request.json()
    clip_url = data.get("clipUrl")
    clip_title = data.get("clipTitle")
    user_id = data.get("userId")  # or extract from session

    if not clip_url or not clip_title or not user_id:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Retrieve access token from DB
    token_record = await tiktok_token_service.get_tokens(user_id)
    if not token_record:
        raise HTTPException(status_code=400, detail="TikTok account not connected")

    access_token = token_record["access_token"]

    # Prepare payload for Content Posting API
    payload = {
        "video_url": clip_url,
        "text": clip_title
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://open.tiktokapis.com/v2/post/publish/video/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )

    if resp.status_code != 200:
        # Include TikTok response body for debugging
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()




# ! Proxy route

def build_proxy_url_from_s3(s3_url: str) -> str:
    """
    Converts S3 URL to a publicly accessible URL under the verified domain
    """
    if ".s3." in s3_url and "amazonaws.com" in s3_url:
        # split to get path after .com/
        if ".com/" in s3_url:
            path = s3_url.split(".com/", 1)[1]
        else:
            path = s3_url.split("/", 3)[-1]

        # include router prefix
        return f"{settings.PROXY_BASE_URL}/tiktok/proxy/{path}"
    return s3_url



@router.get("/proxy/{file_path:path}")
async def proxy_video(file_path: str, request: Request):
    """
    Streams the S3 object located at:
      https://pod-clip-bucket.s3.us-east-2.amazonaws.com/{file_path}
    Example incoming: /proxy/outputs/d1fc.../clip_clip_0.mp4
    """
    s3_url = f"https://pod-clip-bucket.s3.us-east-2.amazonaws.com/{file_path}"
    logger.info(f"Proxy request for: {s3_url}")

    # Use streaming to avoid loading large file into memory
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # stream=True returns an async iterator of chunks
            resp = await client.get(s3_url, timeout=60.0)
        except httpx.HTTPError as e:
            logger.error(f"Error fetching from S3: {e}")
            raise HTTPException(status_code=502, detail="Failed to fetch video from origin")

        if resp.status_code != 200:
            logger.warning(f"S3 returned non-200 ({resp.status_code}) for {s3_url}")
            raise HTTPException(status_code=resp.status_code, detail="Origin returned error")

        # Determine content-type if present (fall back to video/mp4)
        content_type = resp.headers.get("content-type", "video/mp4")
        headers = {
            "Content-Type": content_type,
            # Tell clients not to cache heavily during testing; change for prod if desired
            "Cache-Control": "no-cache, no-store, must-revalidate",
        }

        # Use generator to stream content
        async def iter_stream():
            async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                yield chunk

        return StreamingResponse(iter_stream(), headers=headers, media_type=content_type)
