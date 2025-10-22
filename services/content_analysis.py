import json
import os

# Import configuration
import sys
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

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

        print("--------------------------------")
        print(transcript)
        print("--------------------------------")

        # Handle empty segments gracefully - this is valid for silent videos
        if not segments:
            print(
                "No transcript segments found - video appears to be silent or have no speech"
            )
            print("Creating default segment for silent video")
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
        print(f"Error in content analysis: {str(e)}")
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

    print("--------------------------------")
    print("segments", segments)
    print("--------------------------------")

    # Combine segments into a single text with timestamps
    transcript_text = ""
    for i, segment in enumerate(segments):
        transcript_text += (
            f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
        )

    print("--------------------------------")
    print("transcript text", transcript_text)
    print("--------------------------------")

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
        - Be at least 3 minutes long (clips can be longer, as long as they remain compelling)
        - Avoid long pauses, filler words, or fragmented thoughts
        - Be valuable or entertaining to people

        DO NOT pick random segments. Only select clips that could realistically go viral or spark curiosity, debate, or learning.

        You can return up to, but not limited to, 10 clips.

        For each selected clip, return a JSON object with:
        - "start_time": in seconds
        - "end_time": in seconds
        - "title": a short, compelling title (4–10 words)

        Respond ONLY with a JSON array of clip objects. Do not include extra commentary or explanations.

        Transcript:
        {transcript_text}
        """

    print("-------------------------------\n", transcript_text)

    # Call OpenAI API
    response = await aclient.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that analyzes podcast transcripts to find engaging segments for social media clips.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=2500,
        n=1,
        stop=None,
    )

    # Extract and parse the response
    try:

        print("----------------------\n")
        print("response", response)
        print("----------------------")

        content = response.choices[0].message.content.strip()

        # Find JSON in the response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            segments = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            segments = json.loads(content)

        # Validate and format segments
        valid_segments = []
        for segment in segments:
            if (
                isinstance(segment, dict)
                and "start_time" in segment
                and "end_time" in segment
            ):
                # Ensure minimum duration
                duration = segment["end_time"] - segment["start_time"]
                if duration >= settings.PREFERRED_CLIP_DURATION:
                    valid_segments.append(
                        {
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "title": segment.get("title", "Engaging Clip"),
                            "description": segment.get("description", ""),
                        }
                    )

        print("\n\n\n--------------------------------")
        print("valid segments:--", valid_segments)
        print("--------------------------------\n\n\n")

        return valid_segments

    except Exception as e:
        print(f"Error parsing OpenAI response: {str(e)}")
        print(
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
        print(f"Error generating title: {str(e)}")
        return "Engaging Clip"
