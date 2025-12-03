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

    # Estimate total video duration from segments
    try:
        total_duration = 0.0
        if segments:
            last_end = max((s.get("end") or s.get("end_time") or 0) for s in segments)
            first_start = min((s.get("start") or s.get("start_time") or 0) for s in segments)
            total_duration = max(0.0, float(last_end) - float(first_start))
        total_minutes = max(1, int(total_duration // 60))
    except Exception:
        total_duration = 0.0
        total_minutes = 0

    # Compute dynamic target clip counts (aiming for many clips on long videos)
    # Rough density: ~1 clip per 2 minutes, clamped to MAX_CLIPS_PER_EPISODE
    rough_target = int(total_duration / 120) if total_duration > 0 else 6
    target_min = max(6 if total_minutes < 20 else 8, min(rough_target, settings.MAX_CLIPS_PER_EPISODE))
    target_max = min(target_min + 10, settings.MAX_CLIPS_PER_EPISODE)

    # Log targets
    logger.info(f"Clip targets (speaker-labeled): duration~{total_minutes}m -> min {target_min}, max {target_max}, cap {settings.MAX_CLIPS_PER_EPISODE}")

    # Enhanced prompt that leverages AssemblyAI's perfect sentence boundaries
    prompt = f"""
You are an expert AI specialized in extracting viral-worthy video clips from speaker-diarized transcripts. You have been given a PERFECTLY FORMATTED transcript where each line represents a complete sentence with precise timestamps and speaker labels:

FORMAT: [START_TIME - END_TIME] SPEAKER_X: complete sentence

VIDEO DURATION: ~{total_minutes} minutes
TARGET: Generate {target_min}-{target_max} exceptional clips (quality over quantity)

═══════════════════════════════════════════════════════════════
🎯 CRITICAL BOUNDARY RULES (MUST FOLLOW):
═══════════════════════════════════════════════════════════════

1. **SENTENCE BOUNDARIES ARE SACRED**:
   - ✅ ALWAYS start at [START_TIME] of a sentence
   - ✅ ALWAYS end at [END_TIME] of a sentence
   - ❌ NEVER start or end mid-sentence
   - ❌ NEVER use arbitrary timestamps between sentences

2. **SELECTING START POINT**:
   - Find the interesting moment
   - Go back to find the FIRST sentence that provides necessary context
   - Use that sentence's START_TIME as your clip start
   - Example: If exciting moment is at [120.5], but setup starts at [115.2], use 115.2

3. **SELECTING END POINT**:
   - From the interesting moment, continue forward to the LAST sentence that completes the thought
   - Use that sentence's END_TIME as your clip end
   - Example: If punchline is at [135.8] but speaker continues to [140.3], use 140.3

4. **SPEAKER TURN AWARENESS**:
   - If a clip involves dialogue, include complete exchanges
   - Don't end mid-conversation - wait for a natural conversational pause
   - Example: 
     ❌ BAD: [100.0 - 110.0] cuts off during SPEAKER_B's response
     ✅ GOOD: [100.0 - 115.5] includes full exchange: A asks, B responds completely

═══════════════════════════════════════════════════════════════
🔥 WHAT MAKES A VIRAL CLIP:
═══════════════════════════════════════════════════════════════

**TIER 1 - MUST INCLUDE** (highest priority):
  • Controversial statements or hot takes
  • Shocking revelations or plot twists
  • Heated debates or disagreements
  • Unexpected punchlines or comebacks
  • "Did they really just say that?" moments

**TIER 2 - HIGHLY VALUABLE**:
  • Funny jokes or witty remarks
  • Emotional stories that give chills
  • Practical life-changing advice
  • Behind-the-scenes secrets
  • Bold predictions or claims

**TIER 3 - GOOD TO INCLUDE**:
  • Interesting facts or statistics
  • Personal anecdotes with lessons
  • "Aha!" moments or insights
  • Topic introductions that hook viewers

═══════════════════════════════════════════════════════════════
📋 SELECTION PROCESS:
═══════════════════════════════════════════════════════════════

STEP 1: READ THE ENTIRE TRANSCRIPT
- Identify ALL potentially viral moments
- Mark moments where you felt: surprised, intrigued, entertained, or informed

STEP 2: PRIORITIZE BY TIER
- Start with Tier 1 moments (controversial, shocking)
- Add Tier 2 moments (funny, emotional)
- Fill remaining slots with Tier 3 (interesting, insightful)

STEP 3: FOR EACH SELECTED MOMENT:
a) Find the sentence containing the key moment
b) Expand backwards: What context is needed? (usual: 1-3 sentences)
c) Expand forwards: Is the thought complete? (usual: 1-2 sentences)
d) Verify: Does this stand alone without prior video knowledge?
e) Check duration: 15-180 seconds? (prefer 20-60s)

STEP 4: APPLY BOUNDARIES
- Use START_TIME of first sentence
- Use END_TIME of last sentence
- NEVER modify timestamps - use exact values from transcript

═══════════════════════════════════════════════════════════════
⚙️ TECHNICAL REQUIREMENTS:
═══════════════════════════════════════════════════════════════

Duration: 15-180 seconds (prefer 20-60 seconds)
Count: {target_min} minimum, {target_max} maximum
Distribution: Spread across beginning (25%), middle (50%), end (25%)
Overlap: Clips should NOT overlap in time
Quality: Every clip should be genuinely share-worthy

═══════════════════════════════════════════════════════════════
📝 EXAMPLES (SHOWING CORRECT BOUNDARY SELECTION):
═══════════════════════════════════════════════════════════════

EXAMPLE 1: Controversial Statement
Transcript:
[100.0 - 103.5] SPEAKER_A: So here's my controversial take.
[103.5 - 108.2] SPEAKER_A: I think social media is actually making us happier, not sadder.
[108.2 - 112.8] SPEAKER_A: Everyone says it's toxic, but I disagree completely.
[112.8 - 115.0] SPEAKER_B: Wait, really?
[115.0 - 120.5] SPEAKER_B: That's a hot take considering all the research.

✅ CORRECT CLIP: {{"start_time": 100.0, "end_time": 120.5, "title": "Social Media Makes Us Happier?"}}
   - Includes setup ("here's my controversial take")
   - Includes full statement
   - Includes reaction (adds context/tension)
   - Uses exact sentence boundaries

❌ WRONG: {{"start_time": 103.5, "end_time": 112.8"}}
   - Missing setup context
   - Cuts off mid-conversation (no reaction)

EXAMPLE 2: Funny Moment
Transcript:
[200.0 - 205.3] SPEAKER_A: I tried to impress my date by cooking.
[205.3 - 210.8] SPEAKER_A: I set off the fire alarm making toast.
[210.8 - 213.5] SPEAKER_B: Toast? You set off the alarm with toast?
[213.5 - 218.2] SPEAKER_A: In my defense, it was very crispy toast.

✅ CORRECT CLIP: {{"start_time": 200.0, "end_time": 218.2, "title": "Fire Alarm From Toast"}}
   - Complete story arc: setup → punchline → reaction → callback
   - Natural ending after final joke

═══════════════════════════════════════════════════════════════
🎬 OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════

Return ONLY a JSON array. Each clip must have:
- start_time: EXACT timestamp from transcript (sentence START_TIME)
- end_time: EXACT timestamp from transcript (sentence END_TIME)  
- title: Compelling, curiosity-inducing title (4-8 words)

Example:
[
  {{"start_time": 100.0, "end_time": 120.5, "title": "Why I Quit My $500K Job"}},
  {{"start_time": 300.2, "end_time": 335.8, "title": "The Hardest Truth About Success"}}
]

═══════════════════════════════════════════════════════════════
📊 TRANSCRIPT TO ANALYZE:
═══════════════════════════════════════════════════════════════

{transcript_text}

═══════════════════════════════════════════════════════════════
REMEMBER:
- Use EXACT timestamps from sentences (don't create new ones)
- NEVER cut mid-sentence or mid-word
- Prioritize viral potential over number of clips
- Each clip should make someone want to watch / share immediately
═══════════════════════════════════════════════════════════════
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
        temperature=0.7,
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

        # Enforce global cap to avoid over-generation
        if len(valid_segments) > settings.MAX_CLIPS_PER_EPISODE:
            logger.info(f"⚠️ Truncating segments to MAX_CLIPS_PER_EPISODE={settings.MAX_CLIPS_PER_EPISODE}")
            valid_segments = valid_segments[:settings.MAX_CLIPS_PER_EPISODE]

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

    # Estimate total duration and compute dynamic targets
    try:
        total_duration = 0.0
        if segments:
            last_end = max((s.get("end") or 0) for s in segments)
            first_start = min((s.get("start") or 0) for s in segments)
            total_duration = max(0.0, float(last_end) - float(first_start))
        total_minutes = max(1, int(total_duration // 60))
    except Exception:
        total_duration = 0.0
        total_minutes = 0

    rough_target = int(total_duration / 120) if total_duration > 0 else 6
    target_min = max(6 if total_minutes < 20 else 8, min(rough_target, settings.MAX_CLIPS_PER_EPISODE))
    target_max = min(target_min + 10, settings.MAX_CLIPS_PER_EPISODE)

    # Log targets
    logger.info(f"Clip targets: duration~{total_minutes}m -> min {target_min}, max {target_max}, cap {settings.MAX_CLIPS_PER_EPISODE}")

    # Create prompt for OpenAI with explicit targets
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
        - Each clip should be between 15-180 seconds long to ensure proper context and engagement
        - Target around 30-60 seconds for optimal social media performance
        - MINIMUM: 15 seconds (clips shorter than this will be REJECTED)
        - MAXIMUM: 180 seconds (3 minutes - for in-depth stories/discussions)
        - Choose duration based on the content: expand timestamps to capture complete thoughts/stories
        - Example: If an interesting moment spans 25 seconds, set start_time and end_time to capture those 25 seconds
        - Example: If a story takes 45 seconds to tell completely, use the full 45 seconds
        - Example: For a quick insight at 500s that takes 35 seconds: start_time=500, end_time=535

        TIMESTAMP CALCULATION GUIDELINES:
        1. Identify the most interesting moment or topic in a section
        2. Find where the topic/story BEGINS (include a bit of context if needed)
        3. Find where the topic/story ENDS (ensure the thought is complete)
        4. Calculate duration = end_time - start_time
        5. Ensure duration is >= 15 seconds AND <= 180 seconds
        6. If a topic is naturally shorter than 15 seconds, expand to include surrounding context
        7. If a topic is longer than 180 seconds, select the most compelling 30-90 second portion

        VIDEO DURATION (approx): ~{total_minutes} minutes
        TARGET OUTPUT: Generate at least {target_min} and up to {target_max} genuinely engaging clips. If you discover more than {target_max} excellent moments, return the best {target_max}. If fewer exist, return as many as truly engaging.
        DISTRIBUTION: Spread clips across the entire video (beginning/middle/end). Avoid clustering in one section.

        IMPORTANT: Generate clips for EVERY genuinely interesting moment you find in the video, prioritizing the {target_min}-{target_max} best moments for this length.
        - Do NOT exceed {target_max} clips in total
        - Quality over quantity - but don't miss any viral-worthy segments
        - Each clip should be genuinely engaging and self-contained
        
        DO NOT pick random segments. Only select clips that could realistically go viral or spark curiosity, debate, or learning.

        For each selected clip, return a JSON object with:
        - "start_time": in seconds (beginning of the interesting segment)
        - "end_time": in seconds (end of the segment - must be at least start_time + 15)
        - "title": a short, compelling title (4–10 words)

        MANDATORY VALIDATION: Before submitting your response, verify EVERY clip meets these requirements:
        - Duration >= 15 seconds (MINIMUM - anything less will be rejected and wasted)
        - Duration <= 180 seconds (MAXIMUM - 3 minutes for in-depth content)
        - Complete thought/story (don't cut off mid-sentence)
        - Self-contained (makes sense without prior context)

        Respond ONLY with a JSON array of clip objects. Do not include extra commentary or explanations. Do not exceed {target_max} items.

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
        temperature=0.7,  # Increased for more creative clip detection
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
        # Enforce global cap to avoid over-generation
        if len(valid_segments) > settings.MAX_CLIPS_PER_EPISODE:
            logger.info(f"⚠️ Truncating segments to MAX_CLIPS_PER_EPISODE={settings.MAX_CLIPS_PER_EPISODE}")
            valid_segments = valid_segments[:settings.MAX_CLIPS_PER_EPISODE]

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