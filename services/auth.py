"""
Authentication service for user management and JWT token handling.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from bson import ObjectId
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.errors import DuplicateKeyError

from config import settings
from database.connection import get_users_collection
from models.user import TokenData, User, UserCreate, UserInDB

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service class"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            
            if user_id is None:
                return None
            
            return TokenData(user_id=user_id, username=username)
        except JWTError:
            return None
    
    @staticmethod
    async def get_user_by_username(username: str) -> Optional[UserInDB]:
        """Get user by username"""
        try:
            users_collection = get_users_collection()
            user_data = users_collection.find_one({"username": username})
            
            if user_data:
                return UserInDB(**user_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[UserInDB]:
        """Get user by email"""
        try:
            users_collection = get_users_collection()
            user_data = users_collection.find_one({"email": email})
            
            if user_data:
                return UserInDB(**user_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        try:
            users_collection = get_users_collection()
            user_data = users_collection.find_one({"_id": ObjectId(user_id)})
            
            if user_data:
                return UserInDB(**user_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
            return None
    
    @staticmethod
    async def create_user(user_create: UserCreate) -> UserInDB:
        """Create a new user"""
        try:
            users_collection = get_users_collection()

            # Validate privacy acceptance
            if not user_create.privacy_accepted:
                raise ValueError("Privacy policy must be accepted")

            # Check if user already exists
            existing_user = await AuthService.get_user_by_username(user_create.username)
            if existing_user:
                raise ValueError("Username already exists")

            existing_email = await AuthService.get_user_by_email(user_create.email)
            if existing_email:
                raise ValueError("Email already exists")

            # Create user document (password is already sanitized by Pydantic)
            hashed_password = AuthService.get_password_hash(user_create.password)
            user_doc = {
                "username": user_create.username,
                "first_name": user_create.first_name,
                "last_name": user_create.last_name,
                "email": user_create.email,
                "hashed_password": hashed_password,
                "is_active": True,
                "privacy_accepted": user_create.privacy_accepted,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }

            # Insert user
            result = users_collection.insert_one(user_doc)
            user_doc["_id"] = result.inserted_id

            return UserInDB(**user_doc)

        except DuplicateKeyError:
            raise ValueError("User already exists")
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user"""
        user = await AuthService.get_user_by_username(username)
        if not user:
            return None
        if not AuthService.verify_password(password, user.hashed_password):
            return None
        return user
    
    @staticmethod
    def user_to_dict(user: UserInDB) -> User:
        """Convert UserInDB to User (for API responses)"""
        return User(
            id=str(user.id),
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            is_active=user.is_active,
            privacy_accepted=user.privacy_accepted,
            created_at=user.created_at,
            updated_at=user.updated_at
        )


# Create global auth service instance
auth_service = AuthService()
