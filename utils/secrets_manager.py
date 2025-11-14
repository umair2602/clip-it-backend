"""
AWS Secrets Manager utility for securely loading environment variables.
Falls back to environment variables if AWS is not available or configured.
"""
import os
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try to import boto3, but don't fail if it's not available
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available, will use environment variables only")


class SecretsManager:
    """Manages loading secrets from AWS Secrets Manager or SSM Parameter Store"""
    
    def __init__(self, region: Optional[str] = None, use_secrets_manager: bool = True):
        """
        Initialize Secrets Manager
        
        Args:
            region: AWS region (defaults to AWS_REGION env var or us-east-1)
            use_secrets_manager: If True, use Secrets Manager; if False, use SSM Parameter Store
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.use_secrets_manager = use_secrets_manager
        self._secrets_client = None
        self._ssm_client = None
        self._secrets_cache: Dict[str, str] = {}
        
        if BOTO3_AVAILABLE:
            try:
                if use_secrets_manager:
                    self._secrets_client = boto3.client('secretsmanager', region_name=self.region)
                else:
                    self._ssm_client = boto3.client('ssm', region_name=self.region)
                logger.info(f"Initialized AWS secrets client for region {self.region}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"Could not initialize AWS client: {e}. Will use environment variables.")
                self._secrets_client = None
                self._ssm_client = None
    
    def get_secret(self, secret_name: str, env_fallback: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value from AWS or environment variables
        
        Args:
            secret_name: Name of the secret (SSM parameter name or Secrets Manager secret name)
            env_fallback: Environment variable name to use as fallback (defaults to secret_name)
        
        Returns:
            Secret value or None if not found
        """
        # First check cache
        if secret_name in self._secrets_cache:
            return self._secrets_cache[secret_name]
        
        # Try environment variable first (for local development)
        env_var_name = env_fallback or secret_name
        env_value = os.getenv(env_var_name)
        if env_value:
            logger.debug(f"Using environment variable for {secret_name}")
            self._secrets_cache[secret_name] = env_value
            return env_value
        
        # Try AWS if available
        if self._secrets_client and self.use_secrets_manager:
            value = self._get_from_secrets_manager(secret_name)
            if value:
                self._secrets_cache[secret_name] = value
                return value
        
        if self._ssm_client:
            value = self._get_from_ssm(secret_name)
            if value:
                self._secrets_cache[secret_name] = value
                return value
        
        logger.warning(f"Secret {secret_name} not found in AWS or environment variables")
        return None
    
    def _get_from_secrets_manager(self, secret_name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self._secrets_client.get_secret_value(SecretId=secret_name)
            secret_string = response.get('SecretString')
            
            # Try to parse as JSON (Secrets Manager often stores JSON)
            try:
                secret_dict = json.loads(secret_string)
                # If it's a dict, try to get the value by the secret name key
                if isinstance(secret_dict, dict):
                    # Common patterns: return the value if there's only one key, or use the secret name
                    if len(secret_dict) == 1:
                        return list(secret_dict.values())[0]
                    # Try to find a key matching the secret name
                    key = secret_name.split('/')[-1].upper().replace('-', '_')
                    return secret_dict.get(key) or secret_dict.get(secret_name)
                return secret_string
            except json.JSONDecodeError:
                return secret_string
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.debug(f"Secret {secret_name} not found in Secrets Manager")
            else:
                logger.warning(f"Error retrieving secret {secret_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error retrieving secret {secret_name}: {e}")
            return None
    
    def _get_from_ssm(self, parameter_name: str) -> Optional[str]:
        """Get secret from SSM Parameter Store"""
        try:
            # Ensure parameter name starts with /
            if not parameter_name.startswith('/'):
                parameter_name = f'/clip-it/{parameter_name}'
            
            response = self._ssm_client.get_parameter(
                Name=parameter_name,
                WithDecryption=True  # Decrypt SecureString parameters
            )
            return response['Parameter']['Value']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ParameterNotFound':
                logger.debug(f"Parameter {parameter_name} not found in SSM")
            else:
                logger.warning(f"Error retrieving parameter {parameter_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error retrieving parameter {parameter_name}: {e}")
            return None
    
    def get_all_secrets(self, secret_prefix: str = "/clip-it/") -> Dict[str, str]:
        """
        Get all secrets with a given prefix from SSM Parameter Store
        
        Args:
            secret_prefix: Prefix to filter parameters (e.g., "/clip-it/")
        
        Returns:
            Dictionary of secret names (without prefix) to values
        """
        secrets = {}
        
        if not self._ssm_client:
            return secrets
        
        try:
            paginator = self._ssm_client.get_paginator('describe_parameters')
            page_iterator = paginator.paginate(
                ParameterFilters=[
                    {
                        'Key': 'Name',
                        'Option': 'BeginsWith',
                        'Values': [secret_prefix]
                    }
                ]
            )
            
            for page in page_iterator:
                for param in page.get('Parameters', []):
                    param_name = param['Name']
                    # Get the actual value
                    value = self._get_from_ssm(param_name)
                    if value:
                        # Remove prefix from key name
                        key = param_name.replace(secret_prefix, '').replace('-', '_').upper()
                        secrets[key] = value
                        self._secrets_cache[param_name] = value
            
            logger.info(f"Loaded {len(secrets)} secrets from SSM with prefix {secret_prefix}")
        except Exception as e:
            logger.warning(f"Error loading all secrets: {e}")
        
        return secrets


# Global instance
_secrets_manager = None

def get_secrets_manager(region: Optional[str] = None, use_secrets_manager: bool = False) -> SecretsManager:
    """Get or create the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(region=region, use_secrets_manager=use_secrets_manager)
    return _secrets_manager

def get_secret(secret_name: str, env_fallback: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret
    
    Args:
        secret_name: Name of the secret
        env_fallback: Environment variable name (defaults to secret_name)
        default: Default value if secret not found
    
    Returns:
        Secret value or default
    """
    manager = get_secrets_manager()
    value = manager.get_secret(secret_name, env_fallback)
    return value if value is not None else default

