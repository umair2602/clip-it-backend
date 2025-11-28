"""
AssemblyAI Transcription Service

Provides transcription with native speaker identification and sentence boundaries.
This replaces Whisper for better clip boundary detection.
"""

import logging
from typing import Dict, List, Any
import assemblyai as aai
from config import settings

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
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speaker_labels=True,          # Enable speaker labels
            punctuate=True,                # Add punctuation
            format_text=True,              # Format text properly
        )
        
        # Create transcriber and transcribe
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        
        # Check for errors
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        logger.info(f"✅ AssemblyAI transcription completed")
        logger.info(f"   Duration: {transcript.audio_duration / 1000:.1f}s")
        logger.info(f"   Confidence: {transcript.confidence * 100:.1f}%")
        
        # Extract full text
        full_text = transcript.text
        
        # Convert words to segments (compatible with existing code)
        segments = []
        for word in transcript.words:
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
        
        if transcript.utterances:
            for utterance in transcript.utterances:
                sentences.append({
                    'start': utterance.start / 1000,
                    'end': utterance.end / 1000,
                    'text': utterance.text,
                    'speaker': utterance.speaker,
                    'confidence': utterance.confidence
                })
                speakers.add(utterance.speaker)
        
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
