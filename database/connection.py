"""
MongoDB connection utility for the Clip-It application.
"""

import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from config import settings

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection manager"""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        
    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(settings.MONGODB_URL)
            self.database = self.client[settings.MONGODB_DB_NAME]
            
            # Test the connection
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str) -> Collection:
        """Get a collection from the database"""
        if not self.database:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except Exception:
            pass
        return False

# Global MongoDB instance
mongodb = MongoDB()

def get_database() -> Database:
    """Get the MongoDB database instance"""
    if mongodb.database is None:
        if not mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
    return mongodb.database

def get_users_collection() -> Collection:
    """Get the users collection"""
    return get_database()["users"]
