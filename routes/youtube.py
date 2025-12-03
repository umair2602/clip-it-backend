
import logging
import os
from typing import Dict, Any, Annotated
import google_auth_oauthlib.flow
import googleapiclient.discovery
import httpx
import io
import asyncio
from fastapi import APIRouter, Request, HTTPException, Depends, status, File, Form, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import StreamingResponse
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload
from pydantic import BaseModel, Field
import tempfile
import ffmpeg
import shutil

from config import settings
from services.auth import auth_service
from services.youtube_tokens import youtube_token_service
from database.connection import get_database
import secrets
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/youtube", tags=["youtube"])


class YouTubeUploadRequest(BaseModel):
    video_url: str = Field(..., description="S3 URL of the video to upload")
    title: str = Field(..., max_length=100, description="Video title")
    description: str = Field("", max_length=5000, description="Video description")
    privacy_status: str = Field("private", description="one of private, public, or unlisted")

class YouTubeUploadResponse(BaseModel):
    video_id: str

security = HTTPBearer()


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

def _extract_user_id(user_like):
    if isinstance(user_like, dict):
        return user_like.get("id") or user_like.get("_id") or user_like.get("user_id")
    return getattr(user_like, "id", None) or getattr(user_like, "_id", None) or getattr(user_like, "user_id", None)

@router.get("/auth-url")
async def youtube_auth_url(current_user: dict = Depends(get_current_user)):
    """
    Redirects the user to Google's OAuth 2.0 consent screen for YouTube.
    """
    try:
        logger.info(f"Using YouTube Redirect URI for auth flow: {settings.YOUTUBE_REDIRECT_URI}")
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config={
                "web": {
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [settings.YOUTUBE_REDIRECT_URI],
                }
            },
            scopes=settings.YOUTUBE_SCOPES,
        )
        flow.redirect_uri = settings.YOUTUBE_REDIRECT_URI

        authorization_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent"
        )

        db = get_database()
        user_id = _extract_user_id(current_user)
        if not user_id:
            raise HTTPException(status_code=400, detail="Could not determine user from token")

        db["oauth_states"].insert_one({
            "state": state,
            "user_id": user_id,
            "provider": "youtube",
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc).timestamp() + 600,
        })

        return JSONResponse({"auth_url": authorization_url})

    except Exception as e:
        logger.error(f"Failed to generate YouTube authorization URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate YouTube authorization URL",
        )

from pydantic import BaseModel, Field
from typing import Optional


class YouTubeProfileResponse(BaseModel):
    """Response model for YouTube profile"""
    email: Optional[str] = None
    name: str
    picture: Optional[str] = None

@router.post("/upload", response_model=YouTubeUploadResponse)
async def upload_to_youtube(
    video_url: str = Form(..., description="S3 URL of the video to upload"),
    title: str = Form(..., max_length=100, description="Video title"),
    description: str = Form("", max_length=5000, description="Video description"),
    privacy_status: str = Form("private", description="one of private, public, or unlisted"),
    cover_file: Optional[UploadFile] = File(None, description="Custom cover image"),
    cover_timestamp_ms: Optional[str] = Form(None, description="Timestamp in ms to extract cover from video"),
    current_user: dict = Depends(get_current_user),
):
    """
    Uploads a video from an S3 URL to the user's YouTube channel.
    Supports custom thumbnail via file or timestamp.
    """
    user_id = _extract_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context")

    credentials = await youtube_token_service.refresh_tokens_if_needed(user_id)
    if not credentials:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="YouTube account not connected.")

    try:
        # 1. Fetch video from S3 URL
        async with httpx.AsyncClient() as client:
            response = await client.get(video_url, follow_redirects=True, timeout=60)
            response.raise_for_status()
            video_content = response.content

        video_file = io.BytesIO(video_content)

        # 2. Build YouTube service
        youtube = googleapiclient.discovery.build('youtube', 'v3', credentials=credentials)

        # 3. Prepare video metadata
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": [],
                "categoryId": "22"  # Default category
            },
            "status": {
                "privacyStatus": privacy_status
            }
        }

        # 4. Perform the upload
        media = MediaIoBaseUpload(video_file, mimetype='video/*', chunksize=1024*1024, resumable=True)
        
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media
        )

        # The googleapiclient is synchronous, so we run it in a thread pool
        loop = asyncio.get_event_loop()
        upload_response = await loop.run_in_executor(None, insert_request.execute)
        
        video_id_uploaded = upload_response["id"]
        logger.info(f"Video uploaded successfully. ID: {video_id_uploaded}")

        # 5. Handle Custom Thumbnail
        thumbnail_path = None
        temp_dir = tempfile.mkdtemp()
        
        try:
            if cover_file:
                # Use provided cover file
                thumbnail_path = os.path.join(temp_dir, f"thumb_{uuid.uuid4()}.jpg")
                with open(thumbnail_path, "wb") as f:
                    shutil.copyfileobj(cover_file.file, f)
                logger.info("Using provided custom cover file.")
                
            elif cover_timestamp_ms:
                # Extract frame from video
                try:
                    timestamp_sec = float(cover_timestamp_ms) / 1000.0
                    
                    # Save video content to temp file for ffmpeg
                    temp_video_path = os.path.join(temp_dir, f"video_{uuid.uuid4()}.mp4")
                    with open(temp_video_path, "wb") as f:
                        f.write(video_content)
                        
                    thumbnail_path = os.path.join(temp_dir, f"extracted_{uuid.uuid4()}.jpg")
                    
                    logger.info(f"Extracting frame at {timestamp_sec}s from video...")
                    (
                        ffmpeg
                        .input(temp_video_path, ss=timestamp_sec)
                        .output(thumbnail_path, vframes=1)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    logger.info("Frame extraction successful.")
                    
                except Exception as e:
                    logger.error(f"Failed to extract frame with ffmpeg: {e}")
                    # Don't fail the whole upload if thumbnail fails
                    thumbnail_path = None

            # Upload thumbnail if we have one
            if thumbnail_path and os.path.exists(thumbnail_path):
                logger.info(f"Uploading thumbnail: {thumbnail_path}")
                try:
                    request_thumb = youtube.thumbnails().set(
                        videoId=video_id_uploaded,
                        media_body=MediaFileUpload(thumbnail_path)
                    )
                    await loop.run_in_executor(None, request_thumb.execute)
                    logger.info("Thumbnail set successfully.")
                except Exception as e:
                    logger.error(f"Failed to set thumbnail: {e}")
                    
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

        return YouTubeUploadResponse(video_id=video_id_uploaded)

    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to download video from S3: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to download video from the provided URL.")
    except Exception as e:
        logger.error(f"Failed to upload video to YouTube: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload video to YouTube: {str(e)}")


@router.get("/me", response_model=YouTubeProfileResponse)
async def get_youtube_profile(current_user: dict = Depends(get_current_user)):
    """
    Get YouTube profile information for the authenticated user.
    """
    user_id = _extract_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context")

    credentials = await youtube_token_service.refresh_tokens_if_needed(user_id)

    if not credentials:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="YouTube account not connected.")

    try:
        service = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        
        return YouTubeProfileResponse(
            email=user_info.get("email"),
            name=user_info.get("name"),
            picture=user_info.get("picture"),
        )
    except Exception as e:
        logger.error(f"Failed to get YouTube profile info: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve YouTube profile.")


@router.post("/disconnect")
async def disconnect_youtube(current_user: dict = Depends(get_current_user)):
    """
    Disconnect YouTube account for the authenticated user.
    """
    user_id = _extract_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context")

    try:
        await youtube_token_service.delete_tokens(user_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "YouTube account disconnected successfully."})
    except Exception as e:
        logger.error(f"Failed to disconnect YouTube account for user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to disconnect YouTube account.")


@router.get("/callback")
async def youtube_callback(request: Request):
    """
    Handles the OAuth 2.0 callback from Google.
    """
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    
    if not state or not code:
        raise HTTPException(status_code=400, detail="Missing state or code from Google OAuth callback")

    db = get_database()
    state_doc = db["oauth_states"].find_one_and_delete({"state": state, "provider": "youtube"})

    if not state_doc:
        raise HTTPException(status_code=400, detail="Invalid or expired state parameter.")

    user_id = state_doc.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Could not associate callback with a user.")

    try:
        flow = google_auth_oauthlib.flow.Flow.from_client_config(
            client_config={
                "web": {
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": [settings.YOUTUBE_REDIRECT_URI],
                }
            },
            scopes=settings.YOUTUBE_SCOPES,
            state=state,
        )
        flow.redirect_uri = settings.YOUTUBE_REDIRECT_URI

        flow.fetch_token(code=code)

        credentials = flow.credentials
        
        await youtube_token_service.save_tokens(user_id, credentials)
        
        youtube_user_email = "Your"
        try:
            service = googleapiclient.discovery.build('oauth2', 'v2', credentials=credentials)
            user_info = service.userinfo().get().execute()
            youtube_user_email = user_info.get('email', 'Your')
        except Exception as e:
            logger.error(f"Could not fetch user info from google: {e}")


        return HTMLResponse(f"""
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Connected</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            text-align: center;
            padding: 60px;
            background: linear-gradient(135deg, #ff4b1f 0%, #ff9068 100%);
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
        <h2>YouTube Connected!</h2>
        <p>{youtube_user_email} account is ready to use.</p>
        <p class="small-text">This window will close automatically in 3 seconds.</p>
    </div>
    <script>
        if (window.opener) {{
            try {{
                window.opener.postMessage({{ type: 'YOUTUBE_AUTH_SUCCESS' }}, '*');
            }} catch (e) {{
                console.log('Could not post message to parent:', e);
            }}
        }}
        
        setTimeout(() => {{
            if (window.opener) {{
                window.close();
            }}
        }}, 3000);
    </script>
</body>
</html>
""")

    except Exception as e:
        logger.error(f"Failed to fetch YouTube tokens: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1 style="color: #ff4444;">❌ OAuth Error</h1>
            <p>An error occurred during YouTube authorization.</p>
            <p><strong>Error:</strong> {str(e)}</p>
            <p style="color: #666; font-size: 14px;">This window will close automatically.</p>
            <script>setTimeout(() => window.close(), 5000);</script>
        </body>
        </html>
        """)
