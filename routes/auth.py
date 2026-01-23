"""
Authentication routes for user registration, login, and token management.
"""

import logging
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models.user import UserCreate, UserLogin, User, Token, RefreshTokenRequest, ForgotPasswordRequest, ResetPasswordRequest, VerifyResetTokenRequest
from services.auth import auth_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme
security = HTTPBearer()


async def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]) -> User:
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
    
    return auth_service.user_to_dict(user)


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user_create: UserCreate):
    """
    Register a new user.
    """
   
    try:
        logger.info(f"[REGISTER] Calling auth_service.create_user()...")
        user_in_db = await auth_service.create_user(user_create)
        logger.info(f"[REGISTER] User created successfully!")
        return auth_service.user_to_dict(user_in_db)
    except ValueError as e:
        logger.error(f"[REGISTER] ValueError: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"[REGISTER] Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/login", response_model=Token)
async def login_user(user_login: UserLogin):
    """
    Authenticate user and return JWT tokens.
    
    - **username**: Username or email
    - **password**: User password
    
    Returns access token and refresh token.
    """
    try:
        # Try to authenticate with username first
        user = await auth_service.authenticate_user(user_login.username, user_login.password)
        
        # If username auth fails, try with email
        if not user:
            user = await auth_service.get_user_by_email(user_login.username)
            if user and auth_service.verify_password(user_login.password, user.hashed_password):
                pass  # User authenticated with email
            else:
                user = None
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username/email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        # Create tokens
        access_token_expires = timedelta(minutes=30)  # 30 minutes
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=access_token_expires
        )
        
        refresh_token = auth_service.create_refresh_token(
            data={"sub": str(user.id), "username": user.username}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token.
    """
    try:
        token_data = auth_service.verify_token(refresh_request.refresh_token, "refresh")
        if token_data is None or token_data.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await auth_service.get_user_by_id(token_data.user_id)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create new tokens
        access_token_expires = timedelta(minutes=30)
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=access_token_expires
        )
        
        refresh_token = auth_service.create_refresh_token(
            data={"sub": str(user.id), "username": user.username}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/me", response_model=User)
async def get_current_user_profile(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Get current user profile.
    
    Requires valid JWT token in Authorization header.
    Returns current user information.
    """
    return current_user


@router.get("/verify")
async def verify_token_endpoint(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Verify if the provided token is valid.
    
    Requires valid JWT token in Authorization header.
    Returns success message if token is valid.
    """
    return {"message": "Token is valid", "user_id": current_user.id, "username": current_user.username}


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """
    Request a password reset email.
    
    - **email**: User's email address
    
    Sends a password reset link to the user's email if the account exists.
    Always returns success to prevent email enumeration attacks.
    """
    try:
        # Import email service here to avoid circular imports
        from services.email_service import email_service
        
        # Find user by email
        user = await auth_service.get_user_by_email(request.email.lower())
        
        if user:
            # Generate reset token
            reset_token = auth_service.create_password_reset_token(
                user_id=str(user.id),
                email=user.email
            )
            
            # Send reset email
            email_sent = await email_service.send_password_reset_email(
                to_email=user.email,
                username=user.username,
                reset_token=reset_token
            )
            
            if email_sent:
                logger.info(f"Password reset email sent to {request.email}")
            else:
                logger.error(f"Failed to send password reset email to {request.email}")
        else:
            # Log but don't reveal that user doesn't exist
            logger.info(f"Password reset requested for non-existent email: {request.email}")
        
        # Always return success to prevent email enumeration
        return {
            "message": "If an account with that email exists, we've sent a password reset link."
        }
        
    except Exception as e:
        logger.error(f"Error in forgot password: {str(e)}", exc_info=True)
        # Still return success to prevent information leakage
        return {
            "message": "If an account with that email exists, we've sent a password reset link."
        }


@router.post("/verify-reset-token")
async def verify_reset_token(request: VerifyResetTokenRequest):
    """
    Verify if a password reset token is valid.
    
    - **token**: Password reset token from email
    
    Returns whether the token is valid and not expired.
    """
    try:
        token_data = auth_service.verify_password_reset_token(request.token)
        
        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Verify user still exists
        user = await auth_service.get_user_by_id(token_data["user_id"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        return {
            "valid": True,
            "email": token_data["email"],
            "message": "Token is valid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying reset token: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )


@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """
    Reset user's password using the reset token.
    
    - **token**: Password reset token from email
    - **new_password**: New password (min 6 characters)
    
    Returns success message on successful password reset.
    """
    try:
        # Import email service here to avoid circular imports
        from services.email_service import email_service
        
        # Verify the token
        token_data = auth_service.verify_password_reset_token(request.token)
        
        if token_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Verify user exists
        user = await auth_service.get_user_by_id(token_data["user_id"])
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        # Update the password
        success = await auth_service.update_user_password(
            user_id=token_data["user_id"],
            new_password=request.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        # Send confirmation email
        await email_service.send_password_changed_confirmation(
            to_email=user.email,
            username=user.username
        )
        
        logger.info(f"Password reset successful for user {user.username}")
        
        return {
            "message": "Password has been reset successfully. You can now sign in with your new password."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting password: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )

