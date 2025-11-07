import json
import logging
import os

# Import configuration
import sys
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Set OpenAI API key
aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def analyze_content(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze transcript to identify engaging segments.

    Args:
        transcript: Transcript from Whisper with segments and timestamps

    Returns:
        List of engaging segments with start and end times
    """
    try:
        # Extract the full text and segments
        full_text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        logger.info(f"Analyzing transcript with {len(segments)} segments")

        # Handle empty segments gracefully - this is valid for silent videos
        if not segments:
            logger.warning(
                "No transcript segments found - video appears to be silent or have no speech"
            )
            logger.info("Creating default segment for silent video")
            # Return a default segment for silent videos (30 seconds from start)
            return [
                {
                    "start_time": 0,
                    "end_time": 30,
                    "title": "Silent Video Clip",
                    "description": "Automatically generated clip from silent or speech-free video",
                }
            ]

        # Prepare segments with timestamps for analysis
        formatted_segments = []
        for segment in segments:
            formatted_segments.append(
                {
                    "text": segment.get("text", ""),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                }
            )

        # Use OpenAI to identify engaging segments
        engaging_segments = await identify_engaging_segments(formatted_segments)

        return engaging_segments

    except Exception as e:
        logger.error(f"Error in content analysis: {str(e)}")
        raise


async def identify_engaging_segments(
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use OpenAI to identify engaging segments in the transcript.

    Args:
        segments: List of transcript segments with timestamps

    Returns:
        List of engaging segments with start and end times
        
    """

    logger.info(f"Identifying engaging segments from {len(segments)} transcript segments")

    # Combine segments into a single text with timestamps
    transcript_text = ""
    for i, segment in enumerate(segments):
        transcript_text += (
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
        )

    logger.debug(f"Transcript text length: {len(transcript_text)} characters")

    # Create prompt for OpenAI
    prompt = f"""
        You are an AI assistant that helps extract short, engaging video clips from podcast audio transcripts. You will be given a raw transcript that contains timestamped segments in the format:

        [START_TIME - END_TIME]  spoken sentence

        Your job is to:
        1. Read through the full transcript.
        2. Identify sections that can work as compelling video clips for social media platforms like TikTok, Instagram Reels, or YouTube Shorts.
        3. Select only the most engaging, self-contained parts. A good clip should:
        - Contain a complete thought, story, or insight that doesn’t rely on external context
        - Be interesting to someone who didn’t hear the rest of the conversation
        - Include opinions, unexpected facts, emotional moments, strong takes, or clever observations
        - Avoid long pauses, filler words, or fragmented thoughts
        - Be valuable or entertaining to people

        CRITICAL DURATION REQUIREMENT - READ CAREFULLY:
        - Each clip should be between 20-60 seconds long to ensure proper context and engagement
        - Target around 30-40 seconds for optimal social media performance
        - MINIMUM: 20 seconds (clips shorter than this will be REJECTED)
        - MAXIMUM: 60 seconds (1 minute - ideal for TikTok, Instagram Reels, YouTube Shorts)
        - Choose duration based on the content: expand timestamps to capture complete thoughts/stories
        - Example: If an interesting moment spans 25 seconds, set start_time and end_time to capture those 25 seconds
        - Example: If a story takes 45 seconds to tell completely, use the full 45 seconds
        - Example: For a quick insight at 500s that takes 35 seconds: start_time=500, end_time=535

        TIMESTAMP CALCULATION GUIDELINES:
        1. Identify the most interesting moment or topic in a section
        2. Find where the topic/story BEGINS (include a bit of context if needed)
        3. Find where the topic/story ENDS (ensure the thought is complete)
        4. Calculate duration = end_time - start_time
        5. Ensure duration is >= 20 seconds AND <= 60 seconds
        6. If a topic is naturally shorter than 20 seconds, expand to include surrounding context
        7. If a topic is longer than 60 seconds, select the most compelling 30-60 second portion

        IMPORTANT: Generate MULTIPLE clips from different parts of the video. Look for as many interesting moments as possible (up to 20 clips).
        Prioritize quality over quantity - each clip should be genuinely engaging and self-contained.
        
        DO NOT pick random segments. Only select clips that could realistically go viral or spark curiosity, debate, or learning.

        For each selected clip, return a JSON object with:
        - "start_time": in seconds (beginning of the interesting segment)
        - "end_time": in seconds (end of the segment - must be at least start_time + 20)
        - "title": a short, compelling title (4–10 words)

        MANDATORY VALIDATION: Before submitting your response, verify EVERY clip meets these requirements:
        - Duration >= 20 seconds (MINIMUM - anything less will be rejected and wasted)
        - Duration <= 60 seconds (MAXIMUM - keeps clips social-media friendly)
        - Complete thought/story (don't cut off mid-sentence)
        - Self-contained (makes sense without prior context)

        Respond ONLY with a JSON array of clip objects. Do not include extra commentary or explanations.

        Transcript:
        {transcript_text}
        """

    # Call OpenAI API
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that analyzes podcast transcripts to find engaging segments for social media clips.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=4000,  # Increased to handle more clips
        n=1,
        stop=None,
    )

    # Extract and parse the response
    try:

        content = response.choices[0].message.content.strip()

        logger.info("=" * 60)
        logger.info("OpenAI Response Content:")
        logger.info(content)
        logger.info("=" * 60)

        # Find JSON in the response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            segments = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            segments = json.loads(content)

        logger.info(f"📊 Parsed {len(segments)} segments from OpenAI")

        # Validate and format segments
        valid_segments = []
        for i, segment in enumerate(segments):
            if (
                isinstance(segment, dict)
                and "start_time" in segment
                and "end_time" in segment
            ):
                # Ensure duration is within acceptable range (20-60 seconds for proper context)
                duration = segment["end_time"] - segment["start_time"]
                logger.info(f"Segment {i}: '{segment.get('title', 'No title')}' - Duration: {duration:.1f}s (start: {segment['start_time']:.1f}, end: {segment['end_time']:.1f})")
                
                if duration >= settings.MIN_CLIP_DURATION and duration <= settings.MAX_CLIP_DURATION:
                    valid_segments.append(
                        {
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "title": segment.get("title", "Engaging Clip"),
                            "description": segment.get("description", ""),
                        }
                    )
                    logger.info(f"  ✅ ACCEPTED (duration: {duration:.1f}s)")
                elif duration > settings.MAX_CLIP_DURATION:
                    # If segment is too long, we could split it into multiple 30-second clips
                    # For now, we'll just skip it and let AI handle the segmentation
                    logger.warning(f"  ❌ REJECTED (too long): {duration:.1f} seconds - Maximum is {settings.MAX_CLIP_DURATION}s")
                else:
                    logger.warning(f"  ❌ REJECTED (too short): {duration:.1f} seconds - Minimum is {settings.MIN_CLIP_DURATION}s")

        logger.info("=" * 60)
        logger.info(f"✅ FINAL RESULT: {len(valid_segments)} valid segments out of {len(segments)} total")
        logger.info("=" * 60)

        return valid_segments

    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
        logger.error(
            f"Response content: {response.choices[0].message.content if response.choices else 'No content'}"
        )
        # Return empty list on error
        return []


async def generate_clip_title(segment_text: str) -> str:
    """Generate a catchy title for a clip using OpenAI.

    Args:
        segment_text: Text content of the segment

    Returns:
        Generated title
    """
    prompt = f"""
    Create a short, catchy title (maximum 50 characters) for a social media clip based on this transcript segment:
    
    "{segment_text}"
    
    The title should be attention-grabbing and relevant to the content.
    """

    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that creates catchy titles for social media clips.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=60,
            n=1,
            stop=None,
        )

        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"').strip("'")

        return title

    except Exception as e:
        logger.error(f"Error generating title: {str(e)}")
        return "Engaging Clip"
