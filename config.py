import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def clean_env_value(value):
    """Clean environment variable value by removing surrounding quotes if present"""
    if value and isinstance(value, str):
        # Check if this looks like the entire .env file content
        if '\\r\\n' in value or '\\n' in value:
            # Try to extract the specific value we need
            if 'TIKTOK_REDIRECT_URI=' in value:
                start = value.find('TIKTOK_REDIRECT_URI=') + len('TIKTOK_REDIRECT_URI=')
                end = value.find('\\r\\n', start)
                if end == -1:
                    end = value.find('\\n', start)
                if end == -1:
                    end = len(value)
                extracted = value[start:end]
                return extracted
            else:
                return value
        
        # Remove single or double quotes from beginning and end
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        
        # Decode URL encoding if present
        if '%' in value:
            try:
                from urllib.parse import unquote
                decoded = unquote(value)
                return decoded
            except:
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
    
    # API keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SIEVE_API_KEY = os.getenv("SIEVE_API_KEY")
    
    # HuggingFace settings (for speaker diarization)
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # AWS S3 settings
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_UPLOAD_PREFIX = "uploads/"
    S3_OUTPUT_PREFIX = "outputs/"
    
    # Processing settings
    MIN_CLIP_DURATION = 15 # Minimum clip duration in seconds (15-180 second range for complete context)
    MAX_CLIP_DURATION = 180 # Maximum clip duration in seconds (3 minutes for in-depth conversations)
    PREFERRED_CLIP_DURATION = 45  # Preferred clip duration in seconds (45 sec for complete thoughts)
    MAX_CLIPS_PER_EPISODE = 20  # Maximum number of clips to extract per episode
    
    # Video settings
    OUTPUT_WIDTH = 1080  # Width for vertical video (9:16 aspect ratio)
    OUTPUT_HEIGHT = 1920  # Height for vertical video (9:16 aspect ratio)
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./podcast_clipper.db")

    # MongoDB settings
    # For local MongoDB: "mongodb://localhost:27017"
    # For MongoDB Atlas: "mongodb+srv://dev:LY5xfiaQW44xju87@clip.76hczqd.mongodb.net/?retryWrites=true&w=majority&appName=clip"
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://dev:LY5xfiaQW44xju87@clip.76hczqd.mongodb.net/?retryWrites=true&w=majority&appName=clip")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "clip_it_db")

    # JWT settings
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 180
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 10

    
    # TikTok settings - Clean any quotes that might be accidentally added
    TIKTOK_CLIENT_KEY = clean_env_value(os.getenv("TIKTOK_CLIENT_KEY", "sbawq3ct99ep10ssn9"))
    TIKTOK_CLIENT_SECRET = clean_env_value(os.getenv("TIKTOK_CLIENT_SECRET", "your_client_secret_here"))
    # Force trailing slash to match console; also expose a no-trailing route
    TIKTOK_REDIRECT_URI = clean_env_value(os.getenv("TIKTOK_REDIRECT_URI", "https://social-viper-accepted.ngrok-free.app/tiktok/callback/"))
    TIKTOK_SCOPES = clean_env_value(os.getenv("TIKTOK_SCOPES", "user.info.basic,user.info.profile,video.publish,video.upload"))
    TIKTOK_API_BASE = clean_env_value(os.getenv("TIKTOK_API_BASE", "https://open.tiktokapis.com/v2"))
    TIKTOK_AUTH_BASE = clean_env_value(os.getenv("TIKTOK_AUTH_BASE", "https://www.tiktok.com/v2"))

# Create settings instance
settings = Settings()

# Create necessary directories
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.OUTPUT_DIR.mkdir(exist_ok=True)