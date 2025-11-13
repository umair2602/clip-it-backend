import json
import logging
import os
import time

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
        transcript: Transcript from Whisper with segments and timestamps.
                   May include 'transcript_for_ai' field with speaker-labeled text.

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

        # Check if we have the AI-ready transcript with speaker labels
        # This is the preferred format for clip generation
        if "transcript_for_ai" in transcript and transcript["transcript_for_ai"]:
            logger.info("Using speaker-labeled transcript (transcript_for_ai) for AI analysis")
            # Use the pre-formatted AI transcript directly
            engaging_segments = await identify_engaging_segments_from_text(
                transcript["transcript_for_ai"],
                segments  # Pass original segments for timestamp reference
            )
        else:
            logger.info("Using basic transcript segments (no speaker labels) for AI analysis")
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


async def identify_engaging_segments_from_text(
    transcript_text: str,
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Use OpenAI to identify engaging segments from pre-formatted transcript text.
    
    This function is used when we have the speaker-labeled transcript ready.
    
    Args:
        transcript_text: Pre-formatted transcript with speaker labels (e.g., transcript_for_ai)
        segments: Original segments for reference
        
    Returns:
        List of engaging segments with start and end times
    """
    logger.info(f"Identifying engaging segments from pre-formatted transcript")
    
    # Log the exact transcript being sent to AI
    logger.info("="*70)
    logger.info("TRANSCRIPT SENT TO AI (for clip generation)")
    logger.info("="*70)
    logger.info(transcript_text)
    logger.info("="*70)

    # Create prompt for OpenAI
    prompt = f"""
        You are an AI assistant that helps extract engaging video clips from podcast audio transcripts. You will be given a transcript with speaker labels and timestamps in the format:

        [START_TIME - END_TIME] SPEAKER_XX: spoken sentence

        Your job is to:
        1. Read through the full transcript carefully.
        2. Identify COMPLETE conversations, stories, or discussions that make compelling standalone clips.
        3. Select engaging content that tells a FULL STORY from beginning to end.
        
        WHAT MAKES A GREAT CLIP:
        - Contains a COMPLETE conversation, story, or idea from START to FINISH
        - Has all necessary context - viewers can understand it without hearing anything before
        - Has a natural conclusion - doesn't end abruptly mid-thought
        - Includes engaging content: stories, debates, explanations, insights, humor, strong opinions
        - For multi-speaker clips: includes the ENTIRE exchange (question AND answer, statement AND response)
        - Stands alone as valuable, entertaining, or informative content

        CRITICAL DURATION REQUIREMENT - READ CAREFULLY:
        - Each clip should capture COMPLETE conversations, stories, or thoughts from START to FINISH
        - MINIMUM: 15 seconds (for quick insights or short exchanges)
        - MAXIMUM: 180 seconds (3 minutes - for in-depth discussions or longer stories)
        - Target range: 30-90 seconds for most clips
        - PRIORITY: Completeness over duration - NEVER cut off mid-conversation or mid-thought
        
        CONVERSATION BOUNDARY DETECTION (USE SPEAKER LABELS):
        - Look for natural conversation boundaries using speaker changes
        - A clip should start when a topic/story BEGINS (even if speaker was mid-sentence before)
        - A clip should end when the topic/story is FULLY RESOLVED or naturally concludes
        - If multiple speakers discuss a topic, include the ENTIRE exchange from first mention to resolution
        - DO NOT cut in the middle of back-and-forth dialogue
        - Example: If SPEAKER_00 asks a question and SPEAKER_01 answers, include BOTH parts
        
        CONTEXT PRESERVATION RULES:
        1. Find where a NEW topic, story, or idea is introduced
        2. Trace that topic through ALL speaker contributions until it naturally concludes
        3. Include setup/context at the beginning (don't start abruptly)
        4. Include conclusion/resolution at the end (don't end abruptly)
        5. For debates/discussions: capture the full exchange, not just one side
        6. For stories: capture from setup through climax to conclusion
        7. For explanations: capture the question AND the complete answer
        
        TIMESTAMP CALCULATION GUIDELINES:
        1. Identify a complete conversational unit (topic introduction → discussion → resolution)
        2. Find the EARLIEST point where this topic is mentioned or introduced
        3. Find the LATEST point where this topic is concluded or naturally transitions
        4. Set start_time at the topic introduction (even if it's 5-10 seconds before the main content)
        5. Set end_time when the topic is FULLY resolved (include any follow-up or reactions)
        6. Duration = end_time - start_time (must be 15-180 seconds)
        7. If a topic naturally takes 120 seconds, use all 120 seconds - don't artificially truncate

        IMPORTANT: Generate MULTIPLE clips from different parts of the video. Look for as many complete, engaging moments as possible (up to 20 clips).
        Prioritize COMPLETENESS and quality over quantity - each clip must tell a full story.
        
        DO NOT pick random segments or partial conversations. Only select clips where the topic has a clear beginning and end.

        For each selected clip, return a JSON object with:
        - "start_time": in seconds (where the topic/conversation STARTS)
        - "end_time": in seconds (where the topic/conversation ENDS completely)
        - "title": a short, compelling title (4–10 words) that describes the COMPLETE content

        MANDATORY VALIDATION: Before submitting your response, verify EVERY clip meets these requirements:
        - Duration >= 15 seconds (MINIMUM - only for complete quick exchanges)
        - Duration <= 180 seconds (MAXIMUM - 3 minutes for in-depth content)
        - COMPLETE conversation/topic from start to finish
        - Includes ALL context needed to understand the clip standalone
        - No abrupt starts or endings - natural entry and exit points
        - If speakers have back-and-forth, include the ENTIRE exchange
        - The clip tells a COMPLETE story, not a fragment

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
                "content": "You are an AI assistant that analyzes podcast transcripts with speaker labels to find engaging segments for social media clips.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=4000,
        n=1,
        stop=None,
    )

    # Parse response
    try:
        response_content = response.choices[0].message.content
        logger.info("OpenAI Response Content:")
        logger.info(response_content)

        # Try to extract JSON from the response
        # Sometimes the model might include markdown code blocks
        import re

        json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, assume the entire response is JSON
            json_str = response_content

        segments = json.loads(json_str)
        
        logger.info(f"📊 Parsed {len(segments)} segments from OpenAI")

        # Validate and filter segments
        valid_segments = []
        for seg in segments:
            duration = seg.get("end_time", 0) - seg.get("start_time", 0)
            
            # Log segment info for debugging
            logger.info(f"Segment: {seg.get('title')} | Duration: {duration:.1f}s | Start: {seg.get('start_time')}s | End: {seg.get('end_time')}s")
            
            if duration < 15:
                logger.warning(
                    f"Skipping segment '{seg.get('title')}' - too short ({duration:.1f}s < 15s minimum)"
                )
                continue
            
            if duration > 180:
                logger.warning(
                    f"Skipping segment '{seg.get('title')}' - too long ({duration:.1f}s > 180s maximum, 3 minutes)"
                )
                continue
            
            valid_segments.append(seg)

        logger.info(f"✅ Validated {len(valid_segments)} segments (filtered out {len(segments) - len(valid_segments)} invalid)")
        
        return valid_segments

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing OpenAI response as JSON: {str(e)}")
        logger.error(f"Response content: {response_content}")
        # Return empty list instead of failing
        return []
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
        return []


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
    
    # Log the exact transcript being sent to AI
    logger.info("="*70)
    logger.info("TRANSCRIPT SENT TO AI (for clip generation)")
    logger.info("="*70)
    logger.info(transcript_text)
    logger.info("="*70)

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
        - Each clip should capture COMPLETE conversations, stories, or thoughts from START to FINISH
        - MINIMUM: 15 seconds (for quick insights or short exchanges)
        - MAXIMUM: 180 seconds (3 minutes - for in-depth discussions or longer stories)
        - Target range: 30-90 seconds for most clips
        - PRIORITY: Completeness over duration - NEVER cut off mid-conversation or mid-thought

        TIMESTAMP CALCULATION GUIDELINES:
        1. Identify a complete conversational unit (topic introduction → discussion → resolution)
        2. Find the EARLIEST point where this topic is mentioned or introduced
        3. Find the LATEST point where this topic is concluded or naturally transitions
        4. Set start_time at the topic introduction (even if it's 5-10 seconds before the main content)
        5. Set end_time when the topic is FULLY resolved (include any follow-up or reactions)
        6. Duration = end_time - start_time (must be 15-180 seconds)
        7. If a topic naturally takes 120 seconds, use all 120 seconds - don't artificially truncate

        IMPORTANT: Generate MULTIPLE clips from different parts of the video. Look for as many interesting moments as possible (up to 20 clips).
        Prioritize COMPLETENESS and quality over quantity - each clip must tell a full story.
        
        DO NOT pick random segments or partial conversations. Only select clips where the topic has a clear beginning and end.

        For each selected clip, return a JSON object with:
        - "start_time": in seconds (where the topic/conversation STARTS)
        - "end_time": in seconds (where the topic/conversation ENDS completely)
        - "title": a short, compelling title (4–10 words) that describes the COMPLETE content

        MANDATORY VALIDATION: Before submitting your response, verify EVERY clip meets these requirements:
        - Duration >= 15 seconds (MINIMUM - only for complete quick exchanges)
        - Duration <= 180 seconds (MAXIMUM - 3 minutes for in-depth content)
        - COMPLETE conversation/topic from start to finish
        - Includes ALL context needed to understand the clip standalone
        - No abrupt starts or endings - natural entry and exit points
        - The clip tells a COMPLETE story, not a fragment

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
