"""
Script to create MongoDB indexes for videos and clips collections
"""
import asyncio
import logging
from database.connection import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_indexes():
    """Create indexes for videos and clips collections"""
    try:
        db = get_database()
        
        # Videos collection indexes
        logger.info("Creating indexes on videos collection...")
        videos_collection = db.videos
        
        # Index on user_id for fast user queries
        videos_collection.create_index("user_id")
        logger.info("✅ Created index: videos.user_id")
        
        # Index on created_at for sorting
        videos_collection.create_index("created_at")
        logger.info("✅ Created index: videos.created_at")
        
        # Compound index for user queries with date sorting
        videos_collection.create_index([("user_id", 1), ("created_at", -1)])
        logger.info("✅ Created compound index: videos.user_id + created_at")
        
        # Clips collection indexes
        logger.info("\nCreating indexes on clips collection...")
        clips_collection = db.clips
        
        # Index on video_id for fast video clip queries
        clips_collection.create_index("video_id")
        logger.info("✅ Created index: clips.video_id")
        
        # Index on user_id for user clip queries
        clips_collection.create_index("user_id")
        logger.info("✅ Created index: clips.user_id")
        
        # Index on created_at for sorting
        clips_collection.create_index("created_at")
        logger.info("✅ Created index: clips.created_at")
        
        # Compound index for video clips with time sorting
        clips_collection.create_index([("video_id", 1), ("start_time", 1)])
        logger.info("✅ Created compound index: clips.video_id + start_time")
        
        logger.info("\n🎉 All indexes created successfully!")
        
        # List all indexes
        logger.info("\n📊 Videos collection indexes:")
        for index in videos_collection.list_indexes():
            logger.info(f"   - {index['name']}: {index.get('key', {})}")
        
        logger.info("\n📊 Clips collection indexes:")
        for index in clips_collection.list_indexes():
            logger.info(f"   - {index['name']}: {index.get('key', {})}")
        
    except Exception as e:
        logger.error(f"❌ Error creating indexes: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(create_indexes())
