import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()
logger = logging.getLogger(__name__)

# Import secrets manager (will use AWS if available, otherwise falls back to env vars)
try:
    from utils.secrets_manager import get_secret

    SECRETS_AVAILABLE = True
except ImportError:
    # If secrets_manager is not available, just use os.getenv
    SECRETS_AVAILABLE = False

    def get_secret(
        secret_name: str, env_fallback: str = None, default: str = None
    ) -> str:
        """Fallback function when secrets_manager is not available"""
        env_var = env_fallback or secret_name
        return os.getenv(env_var, default)


def clean_env_value(value):
    """Clean environment variable value by removing surrounding quotes if present"""
    if value and isinstance(value, str):
        # Check if this looks like the entire .env file content
        if "\\r\\n" in value or "\\n" in value:
            # Try to extract the specific value we need
            if "TIKTOK_REDIRECT_URI=" in value:
                start = value.find("TIKTOK_REDIRECT_URI=") + len("TIKTOK_REDIRECT_URI=")
                end = value.find("\\r\\n", start)
                if end == -1:
                    end = value.find("\\n", start)
                if end == -1:
                    end = len(value)
                extracted = value[start:end]
                return extracted
            else:
                return value

        # Remove single or double quotes from beginning and end
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]

        # Decode URL encoding if present
        if "%" in value:
            try:
                from urllib.parse import unquote

                decoded = unquote(value)
                return decoded
            except Exception:
                return value

        return value
    else:
        return value


class Settings:
    # API settings
    API_TITLE = "AI Podcast Clipper API"
    API_VERSION = "0.1.0"

    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploads"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # API keys - Try AWS SSM first, then environment variables
    OPENAI_API_KEY = get_secret("/clip-it/openai-api-key", "OPENAI_API_KEY")
    SIEVE_API_KEY = get_secret("/clip-it/sieve-api-key", "SIEVE_API_KEY")
    HF_TOKEN = get_secret("/clip-it/hf-token", "HF_TOKEN")
    HUGGINGFACE_TOKEN = get_secret("/clip-it/huggingface-token", "HUGGINGFACE_TOKEN")
    ASSEMBLYAI_API_KEY = get_secret("/clip-it/assemblyai-api-key", "ASSEMBLYAI_API_KEY")
    RAPIDAPI_YOUTUBE_KEY = get_secret("/clip-it/rapidapi-youtube-key", "RAPIDAPI_YOUTUBE_KEY")
    ZYLA_API_KEY = get_secret("/clip-it/zyla-api-key", "ZYLA_API_KEY")

    # AWS S3 settings - Note: AWS credentials should use IAM roles in production
    AWS_ACCESS_KEY_ID = get_secret("/clip-it/aws-access-key-id", "AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = get_secret(
        "/clip-it/aws-secret-access-key", "AWS_SECRET_ACCESS_KEY"
    )
    AWS_REGION = get_secret("/clip-it/aws-region", "AWS_REGION", "us-east-1")
    S3_BUCKET = get_secret("/clip-it/s3-bucket", "S3_BUCKET")
    S3_UPLOAD_PREFIX = "uploads/"
    S3_OUTPUT_PREFIX = "outputs/"

    # Processing settings
    MIN_CLIP_DURATION = 15  # Minimum clip duration in seconds (15-180 second range for complete context)
    MAX_CLIP_DURATION = (
        180  # Maximum clip duration in seconds (3 minutes for in-depth conversations)
    )
    PREFERRED_CLIP_DURATION = (
        45  # Preferred clip duration in seconds (45 sec for complete thoughts)
    )
    MAX_CLIPS_PER_EPISODE = (
        10  # Limit to 10 clips per episode for quality over quantity
    )

    # Video settings
    OUTPUT_WIDTH = 1080  # Width for vertical video (9:16 aspect ratio)
    OUTPUT_HEIGHT = 1920  # Height for vertical video (9:16 aspect ratio)

    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./podcast_clipper.db")

    # MongoDB settings
    # For local MongoDB: "mongodb://localhost:27017"
    # For MongoDB Atlas: "mongodb+srv://dev:LY5xfiaQW44xju87@clip.76hczqd.mongodb.net/?retryWrites=true&w=majority&appName=clip"
    MONGODB_URL = get_secret("/clip-it/mongodb-url", "MONGODB_URL")
    MONGODB_DB_NAME = get_secret(
        "/clip-it/mongodb-db-name", "MONGODB_DB_NAME", "clip_it_db"
    )

    # JWT settings
    JWT_SECRET_KEY = get_secret(
        "/clip-it/jwt-secret-key",
        "JWT_SECRET_KEY",
        "your-super-secret-jwt-key-change-this-in-production",
    )
    JWT_ALGORITHM = "HS256"
    # Make expirations configurable via env; increase sensible defaults
    # Access token default: 24 hours (1440 minutes)
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
    )
    # Refresh token default: 30 days
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")
    )

    # TikTok settings - Clean any quotes that might be accidentally added
    TIKTOK_CLIENT_KEY = clean_env_value(
        get_secret(
            "/clip-it/tiktok-client-key", "TIKTOK_CLIENT_KEY", "sbawq3ct99ep10ssn9"
        )
    )
    TIKTOK_CLIENT_SECRET = clean_env_value(
        get_secret(
            "/clip-it/tiktok-client-secret",
            "TIKTOK_CLIENT_SECRET",
            "your_client_secret_here",
        )
    )
    TIKTOK_VERIFICATION_KEY = clean_env_value(
        get_secret("/clip-it/tiktok-verification-key", "TIKTOK_VERIFICATION_KEY")
    )
    PROXY_BASE_URL = clean_env_value(
        get_secret("/clip-it/proxy-base-url", "PROXY_BASE_URL")
    )
    # Force trailing slash to match console; also expose a no-trailing route
    TIKTOK_REDIRECT_URI = clean_env_value(
        get_secret(
            "/clip-it/tiktok-redirect-uri",
            "TIKTOK_REDIRECT_URI",
            "https://social-viper-accepted.ngrok-free.app/tiktok/callback/",
        )
    )
    TIKTOK_SCOPES = clean_env_value(
        get_secret(
            "/clip-it/tiktok-scopes",
            "TIKTOK_SCOPES",
            "user.info.basic,video.publish,video.upload",
        )
    )
    TIKTOK_API_BASE = clean_env_value(
        get_secret(
            "/clip-it/tiktok-api-base",
            "TIKTOK_API_BASE",
            "https://open.tiktokapis.com/v2",
        )
    )
    TIKTOK_AUTH_BASE = clean_env_value(
        get_secret(
            "/clip-it/tiktok-auth-base", "TIKTOK_AUTH_BASE", "https://www.tiktok.com/v2"
        )
    )

    # Speaker diarization
    SPEAKER_DIARIZATION_ENABLED = (
        os.getenv("SPEAKER_DIARIZATION_ENABLED", "false").lower() == "true"
    )

    # Proxy base URL

    PROXY_BASE_URL = clean_env_value(
        os.getenv("PROXY_BASE_URL", "https://895a753eda46.ngrok-free.app")
    )

    # youtube
    YOUTUBE_CLIENT_ID = clean_env_value(
        get_secret(
            "/clip-it/youtube-client-id",
            "YOUTUBE_CLIENT_ID",
            "784400682902-masbc207h16jkjf6hhlbdap8pkcn5p2i.apps.googleusercontent.com",
        )
    )
    YOUTUBE_CLIENT_SECRET = clean_env_value(
        get_secret(
            "/clip-it/youtube-client-secret",
            "YOUTUBE_CLIENT_SECRET",
            "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",
        )
    )
    YOUTUBE_REDIRECT_URI = clean_env_value(
        get_secret(
            "/clip-it/youtube-redirect-uri",
            "YOUTUBE_REDIRECT_URI",
            "https://api.klipz.ai/youtube/callback",
        )
    )
    YOUTUBE_SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/youtube.readonly",
    ]

    # instagram
    INSTA_APP_ID = clean_env_value(
        get_secret("/clip-it/instagram-app-id", "INSTA_APP_ID", "")
    )
    INSTA_APP_SECRET = clean_env_value(
        get_secret("/clip-it/instagram-app-secret", "INSTA_APP_SECRET", "")
    )
    INSTA_REDIRECT_URI = clean_env_value(
        get_secret("/clip-it/instagram-redirect-uri", "INSTA_REDIRECT_URI", "")
    )
    INSTA_APP_SCOPE = clean_env_value(
        get_secret("/clip-it/instagram-app-scope", "INSTA_APP_SCOPE", "")
    )


# Create settings instance
settings = Settings()


# Create necessary directories
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.OUTPUT_DIR.mkdir(exist_ok=True)
