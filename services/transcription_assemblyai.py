"""
AssemblyAI Transcription Service

Provides transcription with native speaker identification and sentence boundaries.
This replaces Whisper for better clip boundary detection.
"""

import logging
from typing import Dict, List, Any
import assemblyai as aai
from config import settings
import time
import os

logger = logging.getLogger(__name__)

# Configure AssemblyAI
aai.settings.api_key = settings.ASSEMBLYAI_API_KEY


async def transcribe_audio_assemblyai(
    audio_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio using AssemblyAI with speaker labels.
    
    Args:
        audio_path: Path to the audio/video file
        **kwargs: Additional options (ignored for compatibility)
    
    Returns:
        Dictionary with transcript data compatible with existing pipeline:
        {
            'text': str,  # Full transcript text
            'segments': List[Dict],  # Word-level segments with timestamps
            'sentences': List[Dict],  # Sentence-level utterances (NEWI!)
            'speakers': List[str],  # List of speaker IDs
            'transcript_for_ai': str  # Formatted transcript for GPT
        }
    """
    try:
        logger.info(f"🎙️ Starting AssemblyAI transcription for: {audio_path}")
        # Log file size and basic info to help diagnose slow uploads
        try:
            file_size = os.path.getsize(audio_path)
            logger.info(f"   File size: {file_size / (1024*1024):.2f} MB")
        except Exception:
            logger.info("   File size: unknown")
        overall_start = time.time()
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speaker_labels=True,          # Enable speaker labels
            punctuate=True,                # Add punctuation
            format_text=True,              # Format text properly
        )
        
        # Create transcriber
        t0 = time.time()
        transcriber = aai.Transcriber()
        logger.info(f"   Transcriber initialized in {time.time()-t0:.2f}s")

        # Start transcription (this handles upload + processing internally)
        t_start_transcribe = time.time()
        logger.info("   Uploading and submitting file to AssemblyAI (this may take some time)...")
        transcript = transcriber.transcribe(audio_path, config=config)
        t_end_transcribe = time.time()
        logger.info(f"   AssemblyAI transcribe() returned in {t_end_transcribe - t_start_transcribe:.2f}s")
        
        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        logger.info(f"✅ AssemblyAI transcription completed (status={getattr(transcript,'status',None)})")
        # Try to log a few useful SDK fields if present
        try:
            audio_duration = getattr(transcript, 'audio_duration', None)
            if audio_duration is not None:
                logger.info(f"   Audio duration (ms): {audio_duration} -> {audio_duration/1000:.1f}s")
        except Exception:
            pass
        try:
            confidence = getattr(transcript, 'confidence', None)
            if confidence is not None:
                logger.info(f"   Confidence: {confidence * 100:.1f}%")
        except Exception:
            pass
        logger.info(f"   Total transcription (upload+process) elapsed: {time.time() - overall_start:.2f}s")
        
        # Extract full text
        t_segments_start = time.time()
        full_text = transcript.text
        
        # Convert words to segments (compatible with existing code)
        segments = []
        # Build word-level segments
        for word in getattr(transcript, 'words', []):
            segments.append({
                'start': word.start / 1000,  # Convert ms to seconds
                'end': word.end / 1000,
                'text': word.text,
                'confidence': word.confidence
            })
        
        # Extract utterances (complete sentences with speaker labels)
        # This is the KEY ADVANTAGE of AssemblyAI - natural sentence boundaries!
        sentences = []
        speakers = set()
        
        # Sentence-level utterances with speaker labels
        for utterance in getattr(transcript, 'utterances', []) or []:
            sentences.append({
                'start': utterance.start / 1000,
                'end': utterance.end / 1000,
                'text': utterance.text,
                'speaker': utterance.speaker,
                'confidence': utterance.confidence
            })
            speakers.add(utterance.speaker)

        t_segments_end = time.time()
        logger.info(f"   Segment/sentence extraction time: {t_segments_end - t_segments_start:.2f}s")
        
        # Format transcript for AI analysis (with speaker labels)
        transcript_for_ai = ""
        for sentence in sentences:
            start_time = sentence['start']
            end_time = sentence['end']
            speaker = sentence['speaker']
            text = sentence['text']
            transcript_for_ai += f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {text}\n"
        
        logger.info(f"📊 Transcription stats:")
        logger.info(f"   Total words: {len(segments)}")
        logger.info(f"   Total sentences: {len(sentences)}")
        logger.info(f"   Speakers detected: {len(speakers)}")
        logger.info(f"   Total transcription end-to-end time: {time.time() - overall_start:.2f}s")
        
        return {
            'text': full_text,
            'segments': segments,
            'sentences': sentences,  #  NEW! Complete sentences with boundaries
            'speakers': list(speakers),
            'transcript_for_ai': transcript_for_ai
        }
        
    except Exception as e:
        logger.error(f"❌ AssemblyAI transcription error: {str(e)}")
        raise


# Alias for compatibility with existing code
transcribe_audio = transcribe_audio_assemblyai
