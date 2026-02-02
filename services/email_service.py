"""
Email service for sending transactional emails using Resend.
"""

import logging
import os
from typing import Optional

import resend

logger = logging.getLogger(__name__)

# Initialize Resend with API key
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "Klipz <noreply@klipz.ai>")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY
else:
    logger.warning("RESEND_API_KEY not configured - email sending will be disabled")


class EmailService:
    """Email service class for sending transactional emails"""
    
    @staticmethod
    def is_configured() -> bool:
        """Check if email service is properly configured"""
        return RESEND_API_KEY is not None
    
    @staticmethod
    async def send_password_reset_email(
        to_email: str,
        username: str,
        reset_token: str
    ) -> bool:
        """
        Send password reset email to user.
        
        Args:
            to_email: User's email address
            username: User's username for personalization
            reset_token: JWT token for password reset
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not EmailService.is_configured():
            logger.error("Email service not configured - cannot send password reset email")
            return False
        
        reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Your Password</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f5;">
            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td align="center" style="padding: 40px 0;">
                        <table role="presentation" style="width: 600px; max-width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="padding: 40px 40px 20px 40px; text-align: center; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); border-radius: 12px 12px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">Klipz</h1>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px;">
                                    <h2 style="margin: 0 0 20px 0; color: #18181b; font-size: 24px; font-weight: 600;">Reset Your Password</h2>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Hi {username},
                                    </p>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        We received a request to reset your password for your Klipz account. Click the button below to create a new password:
                                    </p>
                                    
                                    <!-- CTA Button -->
                                    <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                        <tr>
                                            <td align="center" style="padding: 20px 0;">
                                                <a href="{reset_link}" style="display: inline-block; padding: 14px 32px; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 8px; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);">
                                                    Reset Password
                                                </a>
                                            </td>
                                        </tr>
                                    </table>
                                    
                                    <p style="margin: 20px 0 0 0; color: #71717a; font-size: 14px; line-height: 1.6;">
                                        This link will expire in <strong>1 hour</strong> for security reasons.
                                    </p>
                                    <p style="margin: 15px 0 0 0; color: #71717a; font-size: 14px; line-height: 1.6;">
                                        If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
                                    </p>
                                    
                                    <!-- Alternative Link -->
                                    <div style="margin-top: 30px; padding: 20px; background-color: #f4f4f5; border-radius: 8px;">
                                        <p style="margin: 0 0 10px 0; color: #71717a; font-size: 13px;">
                                            If the button doesn't work, copy and paste this link into your browser:
                                        </p>
                                        <p style="margin: 0; color: #8B5CF6; font-size: 13px; word-break: break-all;">
                                            {reset_link}
                                        </p>
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px 40px; border-top: 1px solid #e4e4e7; text-align: center;">
                                    <p style="margin: 0; color: #a1a1aa; font-size: 13px;">
                                        © 2024 Klipz. All rights reserved.
                                    </p>
                                    <p style="margin: 10px 0 0 0; color: #a1a1aa; font-size: 13px;">
                                        Questions? Contact us at support@klipz.ai
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        text_content = f"""
Reset Your Password

Hi {username},

We received a request to reset your password for your Klipz account.

Click the link below to create a new password:
{reset_link}

This link will expire in 1 hour for security reasons.

If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.

---
© 2024 Klipz. All rights reserved.
Questions? Contact us at support@klipz.ai
        """
        
        try:
            params: resend.Emails.SendParams = {
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Reset Your Password - Klipz",
                "html": html_content,
                "text": text_content,
            }
            
            email_response = resend.Emails.send(params)
            logger.info(f"Password reset email sent successfully to {to_email}, id: {email_response.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send password reset email to {to_email}: {str(e)}")
            return False
    
    @staticmethod
    async def send_password_changed_confirmation(
        to_email: str,
        username: str
    ) -> bool:
        """
        Send confirmation email after password has been changed.
        
        Args:
            to_email: User's email address
            username: User's username for personalization
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not EmailService.is_configured():
            logger.error("Email service not configured - cannot send confirmation email")
            return False
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password Changed Successfully</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f5;">
            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td align="center" style="padding: 40px 0;">
                        <table role="presentation" style="width: 600px; max-width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="padding: 40px 40px 20px 40px; text-align: center; background: linear-gradient(135deg, #10B981 0%, #059669 100%); border-radius: 12px 12px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">Klipz</h1>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px;">
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <span style="display: inline-block; width: 60px; height: 60px; background-color: #D1FAE5; border-radius: 50%; line-height: 60px; font-size: 30px;">✓</span>
                                    </div>
                                    <h2 style="margin: 0 0 20px 0; color: #18181b; font-size: 24px; font-weight: 600; text-align: center;">Password Changed Successfully</h2>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Hi {username},
                                    </p>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Your password has been successfully changed. You can now sign in with your new password.
                                    </p>
                                    <p style="margin: 20px 0 0 0; color: #71717a; font-size: 14px; line-height: 1.6;">
                                        If you did not make this change, please contact our support team immediately at <a href="mailto:support@klipz.ai" style="color: #8B5CF6;">support@klipz.ai</a>.
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px 40px; border-top: 1px solid #e4e4e7; text-align: center;">
                                    <p style="margin: 0; color: #a1a1aa; font-size: 13px;">
                                        © 2024 Klipz. All rights reserved.
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        text_content = f"""
Password Changed Successfully

Hi {username},

Your password has been successfully changed. You can now sign in with your new password.

If you did not make this change, please contact our support team immediately at support@klipz.ai.

---
© 2024 Klipz. All rights reserved.
        """
        
        try:
            params: resend.Emails.SendParams = {
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Password Changed Successfully - Klipz",
                "html": html_content,
                "text": text_content,
            }
            
            email_response = resend.Emails.send(params)
            logger.info(f"Password changed confirmation sent to {to_email}, id: {email_response.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send password changed confirmation to {to_email}: {str(e)}")
            return False


    @staticmethod
    async def send_welcome_email(
        to_email: str,
        username: str,
        first_name: str
    ) -> bool:
        """
        Send welcome email to newly registered user.
        
        Args:
            to_email: User's email address
            username: User's username
            first_name: User's first name for personalization
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not EmailService.is_configured():
            logger.error("Email service not configured - cannot send welcome email")
            return False
        
        login_link = f"{FRONTEND_URL}/sign-in"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to Klipz!</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f5;">
            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td align="center" style="padding: 40px 0;">
                        <table role="presentation" style="width: 600px; max-width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="padding: 40px 40px 20px 40px; text-align: center; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); border-radius: 12px 12px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">Klipz</h1>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px;">
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <span style="display: inline-block; width: 60px; height: 60px; background-color: #EDE9FE; border-radius: 50%; line-height: 60px; font-size: 30px;">🎉</span>
                                    </div>
                                    <h2 style="margin: 0 0 20px 0; color: #18181b; font-size: 24px; font-weight: 600; text-align: center;">Welcome to Klipz!</h2>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Hi {first_name},
                                    </p>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Thank you for creating your Klipz account! We're excited to have you on board.
                                    </p>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        With Klipz, you can easily create stunning video clips powered by AI. Here's what you can do:
                                    </p>
                                    <ul style="margin: 0 0 20px 0; padding-left: 20px; color: #52525b; font-size: 16px; line-height: 1.8;">
                                        <li>Upload your videos and let AI find the best moments</li>
                                        <li>Automatically generate engaging clips for social media</li>
                                        <li>Share directly to TikTok, YouTube, and more</li>
                                    </ul>
                                    
                                    <!-- CTA Button -->
                                    <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                        <tr>
                                            <td align="center" style="padding: 20px 0;">
                                                <a href="{login_link}" style="display: inline-block; padding: 14px 32px; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 8px; box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);">
                                                    Get Started
                                                </a>
                                            </td>
                                        </tr>
                                    </table>
                                    
                                    <div style="margin-top: 30px; padding: 20px; background-color: #f4f4f5; border-radius: 8px;">
                                        <p style="margin: 0 0 10px 0; color: #52525b; font-size: 14px; font-weight: 600;">
                                            Your Account Details:
                                        </p>
                                        <p style="margin: 0; color: #71717a; font-size: 14px;">
                                            Username: <strong>{username}</strong><br>
                                            Email: <strong>{to_email}</strong>
                                        </p>
                                    </div>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px 40px; border-top: 1px solid #e4e4e7; text-align: center;">
                                    <p style="margin: 0; color: #a1a1aa; font-size: 13px;">
                                        © 2024 Klipz. All rights reserved.
                                    </p>
                                    <p style="margin: 10px 0 0 0; color: #a1a1aa; font-size: 13px;">
                                        Questions? Contact us at support@klipz.ai
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        text_content = f"""
Welcome to Klipz!

Hi {first_name},

Thank you for creating your Klipz account! We're excited to have you on board.

With Klipz, you can easily create stunning video clips powered by AI. Here's what you can do:

- Upload your videos and let AI find the best moments
- Automatically generate engaging clips for social media
- Share directly to TikTok, YouTube, and more

Get started here: {login_link}

Your Account Details:
Username: {username}
Email: {to_email}

---
© 2024 Klipz. All rights reserved.
Questions? Contact us at support@klipz.ai
        """
        
        try:
            params: resend.Emails.SendParams = {
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": "Welcome to Klipz! 🎉",
                "html": html_content,
                "text": text_content,
            }
            
            email_response = resend.Emails.send(params)
            logger.info(f"Welcome email sent successfully to {to_email}, id: {email_response.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {to_email}: {str(e)}")
            return False

    @staticmethod
    async def send_job_completed_email(
        to_email: str,
        first_name: str,
        video_title: str,
        clips_count: int,
        video_id: str
    ) -> bool:
        """
        Send email notification when video processing job is completed.
        
        Args:
            to_email: User's email address
            first_name: User's first name for personalization
            video_title: Title of the processed video
            clips_count: Number of clips generated
            video_id: ID of the processed video
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not EmailService.is_configured():
            logger.error("Email service not configured - cannot send job completed email")
            return False
        
        history_link = f"{FRONTEND_URL}/history"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your Clips Are Ready!</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f5;">
            <table role="presentation" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td align="center" style="padding: 40px 0;">
                        <table role="presentation" style="width: 600px; max-width: 100%; border-collapse: collapse; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="padding: 40px 40px 20px 40px; text-align: center; background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%); border-radius: 12px 12px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">Klipz</h1>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px;">
                                    <div style="text-align: center; margin-bottom: 20px;">
                                        <span style="display: inline-block; width: 60px; height: 60px; background-color: #D1FAE5; border-radius: 50%; line-height: 60px; font-size: 30px;">🎬</span>
                                    </div>
                                    <h2 style="margin: 0 0 20px 0; color: #18181b; font-size: 24px; font-weight: 600; text-align: center;">Your Clips Are Ready!</h2>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Hi {first_name},
                                    </p>
                                    <p style="margin: 0 0 20px 0; color: #52525b; font-size: 16px; line-height: 1.6;">
                                        Great news! Your video has been processed successfully and your clips are ready to view.
                                    </p>
                                    
                                    <!-- Video Info Box -->
                                    <div style="margin: 25px 0; padding: 20px; background-color: #f4f4f5; border-radius: 8px; border-left: 4px solid #10B981;">
                                        <p style="margin: 0 0 10px 0; color: #18181b; font-size: 16px; font-weight: 600;">
                                            📹 {video_title}
                                        </p>
                                        <p style="margin: 0; color: #52525b; font-size: 14px;">
                                            <strong>{clips_count}</strong> clip{'s' if clips_count != 1 else ''} generated
                                        </p>
                                    </div>
                                    
                                    <!-- CTA Button -->
                                    <table role="presentation" style="width: 100%; border-collapse: collapse;">
                                        <tr>
                                            <td align="center" style="padding: 20px 0;">
                                                <a href="{history_link}" style="display: inline-block; padding: 14px 32px; background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; border-radius: 8px; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);">
                                                    View Your Clips
                                                </a>
                                            </td>
                                        </tr>
                                    </table>
                                    
                                    <p style="margin: 20px 0 0 0; color: #71717a; font-size: 14px; line-height: 1.6; text-align: center;">
                                        You can now download your clips or share them directly to TikTok, YouTube, and Instagram!
                                    </p>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px 40px; border-top: 1px solid #e4e4e7; text-align: center;">
                                    <p style="margin: 0; color: #a1a1aa; font-size: 13px;">
                                        © 2024 Klipz. All rights reserved.
                                    </p>
                                    <p style="margin: 10px 0 0 0; color: #a1a1aa; font-size: 13px;">
                                        Questions? Contact us at support@klipz.ai
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        text_content = f"""
Your Clips Are Ready!

Hi {first_name},

Great news! Your video has been processed successfully and your clips are ready to view.

Video: {video_title}
Clips Generated: {clips_count}

View your clips here: {history_link}

You can now download your clips or share them directly to TikTok, YouTube, and Instagram!

---
© 2024 Klipz. All rights reserved.
Questions? Contact us at support@klipz.ai
        """
        
        try:
            params: resend.Emails.SendParams = {
                "from": RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": f"🎬 Your Clips Are Ready! - {video_title}",
                "html": html_content,
                "text": text_content,
            }
            
            email_response = resend.Emails.send(params)
            logger.info(f"Job completed email sent successfully to {to_email}, id: {email_response.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send job completed email to {to_email}: {str(e)}")
            return False


# Create global email service instance
email_service = EmailService()
