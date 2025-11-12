"""
Speaker diarization service using pyannote.audio.
Identifies and labels different speakers in audio/video files.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class SpeakerDiarization:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize speaker diarization service.

        Args:
            hf_token: HuggingFace API token for accessing pyannote models.
                     Required for first-time model download.
                     Get it from: https://huggingface.co/settings/tokens
                     And accept terms at: https://huggingface.co/pyannote/speaker-diarization
        """
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the pyannote diarization pipeline"""
        try:
            from pyannote.audio import Pipeline

            if not self.hf_token:
                logger.warning(
                    "No HuggingFace token provided. Speaker diarization will not be available. "
                    "Get a token from https://huggingface.co/settings/tokens and set HF_TOKEN env variable."
                )
                return

            logger.info("Loading pyannote speaker diarization pipeline...")

            # Load the pre-trained pipeline
            # This requires accepting the user agreement at:
            # https://huggingface.co/pyannote/speaker-diarization-3.1
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )

            # Move to GPU if available
            import torch
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Speaker diarization pipeline loaded on GPU")
            else:
                logger.info("Speaker diarization pipeline loaded on CPU")

        except Exception as e:
            logger.error(f"Error loading speaker diarization pipeline: {str(e)}")
            logger.error(
                "Make sure you have:\n"
                "1. Accepted the user agreement at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Set your HuggingFace token in HF_TOKEN environment variable\n"
                "3. Installed pyannote.audio: pip install pyannote.audio"
            )
            self.pipeline = None

    def is_available(self) -> bool:
        """Check if speaker diarization is available"""
        return self.pipeline is not None

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of speaker segments with format:
            [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.5,
                    "end": 3.2
                },
                ...
            ]
        """
        if not self.is_available():
            logger.warning("Speaker diarization not available - skipping")
            return None

        try:
            logger.info(f"Starting speaker diarization for: {audio_path}")

            # Ensure input is a WAV file that pyannote decoders handle reliably
            prepared_path, tmp_dir = self._prepare_audio_for_diarization(audio_path)
            if prepared_path != audio_path:
                logger.info(f"Using converted audio for diarization: {prepared_path}")

            # Prepare diarization parameters
            diarization_params = {}
            if num_speakers is not None:
                diarization_params["num_speakers"] = num_speakers
            if min_speakers is not None:
                diarization_params["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_params["max_speakers"] = max_speakers

            # Optional env-based hints (useful for speed and stability on CPU)
            if "num_speakers" not in diarization_params:
                try:
                    env_num = os.getenv("DIARIZATION_NUM_SPEAKERS")
                    if env_num:
                        diarization_params["num_speakers"] = int(env_num)
                        logger.info(f"Diarization hint: num_speakers={diarization_params['num_speakers']} (from env)")
                except Exception:
                    pass
            if "min_speakers" not in diarization_params:
                try:
                    env_min = os.getenv("DIARIZATION_MIN_SPEAKERS")
                    if env_min:
                        diarization_params["min_speakers"] = int(env_min)
                        logger.info(f"Diarization hint: min_speakers={diarization_params['min_speakers']} (from env)")
                except Exception:
                    pass
            if "max_speakers" not in diarization_params:
                try:
                    env_max = os.getenv("DIARIZATION_MAX_SPEAKERS")
                    if env_max:
                        diarization_params["max_speakers"] = int(env_max)
                        logger.info(f"Diarization hint: max_speakers={diarization_params['max_speakers']} (from env)")
                except Exception:
                    pass

            # Prefer passing waveform directly to avoid backend decoder issues
            try:
                import soundfile as sf
                import torch

                data, sr = sf.read(prepared_path, always_2d=False)
                if data is None or len(data) == 0:
                    raise ValueError("Failed to read audio data for diarization")

                # Ensure (channels, samples) float32 tensor
                if data.ndim == 1:  # mono (samples,)
                    waveform = torch.from_numpy(data).unsqueeze(0).float()
                else:  # (samples, channels) -> (channels, samples)
                    import numpy as np
                    if data.shape[0] < data.shape[1]:
                        # likely (samples, channels)
                        data = np.transpose(data)
                    waveform = torch.from_numpy(data).float()

                logger.info(f"Running diarization with in-memory waveform @ {sr} Hz, shape {tuple(waveform.shape)}")
                diarization = self.pipeline({"waveform": waveform, "sample_rate": int(sr)}, **diarization_params)
            except Exception as decode_err:
                logger.warning(f"Waveform path failed ({decode_err}). Falling back to file path.")
                diarization = self.pipeline(prepared_path, **diarization_params)

            # Convert diarization output to a common list-of-dicts format
            speaker_segments = self._convert_diarization_to_segments(diarization)

            logger.info(
                f"Diarization complete. Found {len(set(s['speaker'] for s in speaker_segments))} "
                f"speakers in {len(speaker_segments)} segments"
            )

            return speaker_segments

        except Exception as e:
            logger.error(f"Error during speaker diarization: {str(e)}")
            return None
        finally:
            # Clean up any temporary conversion artifacts
            try:
                if 'tmp_dir' in locals() and tmp_dir and os.path.isdir(tmp_dir):
                    for p in Path(tmp_dir).glob("*"):
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    try:
                        Path(tmp_dir).rmdir()
                    except Exception:
                        pass
            except Exception:
                pass

    def _prepare_audio_for_diarization(self, audio_path: str) -> Tuple[str, Optional[str]]:
        """
        Convert input audio to 16kHz mono WAV if needed so pyannote can decode it.

        Many Windows/PyTorch builds cannot decode MP3/ACC via torchaudio/soundfile.
        Converting to WAV ensures compatibility and avoids decoder errors like
        "name 'AudioDecoder' is not defined" from underlying backends.

        Returns:
            (prepared_path, tmp_dir) where prepared_path is the path to use for
            diarization and tmp_dir (if not None) should be cleaned after use.
        """
        try:
            ext = Path(audio_path).suffix.lower()
            if ext == ".wav":
                return audio_path, None

            # Convert to temporary WAV (16k mono)
            tmp_dir = tempfile.mkdtemp(prefix="diarize_")
            wav_path = str(Path(tmp_dir) / "audio_16k_mono.wav")

            try:
                import ffmpeg  # ffmpeg-python

                # Optional trimming for faster tests: DIARIZATION_MAX_SECONDS
                trim_seconds = None
                try:
                    env_trim = os.getenv("DIARIZATION_MAX_SECONDS")
                    if env_trim:
                        trim_seconds = int(env_trim)
                        logger.info(f"Diarization trim enabled: first {trim_seconds}s (env DIARIZATION_MAX_SECONDS)")
                except Exception:
                    trim_seconds = None

                out_kwargs = {"ac": 1, "ar": 16000, "format": "wav"}
                if trim_seconds and trim_seconds > 0:
                    out_kwargs["t"] = trim_seconds

                (
                    ffmpeg
                    .input(audio_path)
                    .output(wav_path, **out_kwargs)
                    .overwrite_output()
                    .run(quiet=True)
                )
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    return wav_path, tmp_dir
                else:
                    logger.warning("FFmpeg conversion produced empty WAV, falling back to original audio")
                    return audio_path, None
            except Exception as conv_err:
                logger.warning(f"FFmpeg conversion to WAV failed: {conv_err}. Proceeding with original audio")
                return audio_path, None
        except Exception as e:
            logger.warning(f"Audio preparation failed: {e}. Proceeding with original audio")
            return audio_path, None

    def _convert_diarization_to_segments(self, diarization: Any) -> List[Dict[str, Any]]:
        """
        Normalize various pyannote 2.x/3.x outputs into a list of dicts:
        [{"speaker": str, "start": float, "end": float}]
        """
        segments: List[Dict[str, Any]] = []

        try:
            # Special handling for pyannote 3.x DiarizeOutput dataclass
            if type(diarization).__name__ == "DiarizeOutput":
                logger.info("Diarization output: detected DiarizeOutput type")
                
                # Always log structure for debugging
                try:
                    attrs = dir(diarization)
                    logger.info(f"DiarizeOutput available attributes: {[a for a in attrs if not a.startswith('_')]}")
                except Exception:
                    pass
                
                # DiarizeOutput has .speaker_diarization attribute (not .annotation)
                ann = getattr(diarization, "speaker_diarization", None)
                logger.info(f"DiarizeOutput.speaker_diarization type: {type(ann)}")
                
                if ann is not None and hasattr(ann, "itertracks"):
                    logger.info("DiarizeOutput: using speaker_diarization.itertracks")
                    for turn, _, speaker in ann.itertracks(yield_label=True):
                        segments.append({
                            "speaker": str(speaker),
                            "start": float(turn.start),
                            "end": float(turn.end),
                        })
                    logger.info(f"DiarizeOutput: extracted {len(segments)} segments from speaker_diarization.itertracks")
                    if segments:
                        return segments
                
                # Some versions may provide .to_annotation()
                if hasattr(diarization, "to_annotation"):
                    try:
                        logger.info("DiarizeOutput: trying to_annotation()")
                        ann2 = diarization.to_annotation()
                        logger.info(f"to_annotation() returned: {type(ann2)}")
                        if hasattr(ann2, "itertracks"):
                            logger.info("DiarizeOutput: using to_annotation().itertracks")
                            for turn, _, speaker in ann2.itertracks(yield_label=True):
                                segments.append({
                                    "speaker": str(speaker),
                                    "start": float(turn.start),
                                    "end": float(turn.end),
                                })
                            logger.info(f"DiarizeOutput: extracted {len(segments)} segments from to_annotation()")
                            if segments:
                                return segments
                    except Exception as to_ann_err:
                        logger.warning(f"DiarizeOutput: to_annotation() failed: {to_ann_err}")
                
                # Fallback: attempt generic 'segments' attribute
                raw_segments = getattr(diarization, "segments", None)
                logger.info(f"DiarizeOutput.segments type: {type(raw_segments)}")
                
                if raw_segments and isinstance(raw_segments, (list, tuple)):
                    logger.info("DiarizeOutput: using raw segments attribute")
                    for item in raw_segments:
                        if isinstance(item, dict):
                            start = float(item.get("start", 0.0))
                            end = float(item.get("end", start))
                            spk = str(item.get("speaker", "SPEAKER_00"))
                            segments.append({"speaker": spk, "start": start, "end": end})
                    if segments:
                        logger.info(f"DiarizeOutput: extracted {len(segments)} segments from raw segments")
                        return segments
                
                # Try repr for debugging
                try:
                    logger.warning(f"DiarizeOutput could not be converted. repr: {repr(diarization)[:500]}")
                except Exception:
                    pass
                    
                # Continue to other heuristics if not returned yet

            # Path A: classic pyannote.core.Annotation API
            if hasattr(diarization, "itertracks"):
                logger.info("Diarization output: using Annotation.itertracks path")
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        "speaker": str(speaker),
                        "start": float(turn.start),
                        "end": float(turn.end),
                    })
                return segments

            # Path B: new pyannote 3.x output with to_annotation()
            if hasattr(diarization, "to_annotation"):
                logger.info("Diarization output: using to_annotation().itertracks path")
                ann = diarization.to_annotation()
                if hasattr(ann, "itertracks"):
                    for turn, _, speaker in ann.itertracks(yield_label=True):
                        segments.append({
                            "speaker": str(speaker),
                            "start": float(turn.start),
                            "end": float(turn.end),
                        })
                        
                    return segments

            # Path C: dict-like or attribute-based segments list
            candidate = None
            if isinstance(diarization, dict) and "segments" in diarization:
                candidate = diarization.get("segments")
            elif hasattr(diarization, "segments"):
                candidate = getattr(diarization, "segments")

            if candidate is not None:
                logger.info("Diarization output: using generic 'segments' iterable path")
                for item in list(candidate):
                    # Accept both dicts and simple tuples
                    if isinstance(item, dict):
                        start = float(item.get("start", 0.0))
                        end = float(item.get("end", start))
                        spk = str(item.get("speaker", "SPEAKER_00"))
                    else:
                        # Unknown tuple-like structure; skip safely
                        continue

                    segments.append({
                        "speaker": spk,
                        "start": start,
                        "end": end,
                    })
                if segments:
                    return segments

            # Path D: best-effort fallback — log type to help debugging
            logger.warning(f"Unknown diarization output type: {type(diarization)}; falling back to empty result")
            try:
                logger.debug(f"Diarization dir: {dir(diarization)}")
            except Exception:
                pass
            return []

        except Exception as conv_err:
            logger.warning(f"Failed to normalize diarization output: {conv_err}")
            return []

    def assign_speakers_to_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]],
        audio_path: Optional[str] = None,
        use_whisperx_alignment: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Assign speaker labels to transcript segments based on temporal overlap.
        
        This method supports two approaches:
        1. WhisperX alignment (recommended): Uses word-level timestamps for precise alignment
        2. Basic overlap (fallback): Uses segment-level overlap calculation

        Args:
            transcript_segments: Whisper transcript segments with 'start', 'end', 'text'
            speaker_segments: Speaker diarization segments with 'speaker', 'start', 'end'
            audio_path: Path to audio file (required for WhisperX alignment)
            use_whisperx_alignment: Whether to use WhisperX for word-level alignment

        Returns:
            Enhanced transcript segments with speaker labels
        """
        if not speaker_segments:
            return transcript_segments

        # Try WhisperX alignment if available and requested
        if use_whisperx_alignment and audio_path:
            try:
                logger.info("Using WhisperX for word-level alignment with speakers")
                return self._assign_speakers_with_whisperx(
                    transcript_segments, 
                    speaker_segments, 
                    audio_path
                )
            except ImportError:
                logger.warning(
                    "WhisperX not available. Install with: pip install whisperx\n"
                    "Falling back to basic overlap-based alignment."
                )
            except Exception as e:
                logger.warning(
                    f"WhisperX alignment failed: {str(e)}\n"
                    f"Falling back to basic overlap-based alignment."
                )

        # Fallback to basic overlap-based alignment
        logger.info("Using basic overlap-based alignment")
        return self._assign_speakers_basic(transcript_segments, speaker_segments)

    def _assign_speakers_with_whisperx(
        self,
        transcript_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]],
        audio_path: str
    ) -> List[Dict[str, Any]]:
        """
        Assign speakers using WhisperX word-level alignment.
        
        This is the recommended method as it provides more accurate alignment
        by using word-level timestamps instead of segment-level overlap.
        
        Based on: https://www.kaggle.com/code/sacrum/whisper-ai-pyannote-transcribing
        """
        import whisperx
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get the language from first segment or default to English
        language = transcript_segments[0].get("language", "en") if transcript_segments else "en"
        
        # Load alignment model
        logger.info(f"Loading WhisperX alignment model for language: {language}")
        align_model, metadata = whisperx.load_align_model(
            language_code=language, 
            device=device
        )
        
        # Prepare transcript in WhisperX format
        whisperx_result = {
            "segments": transcript_segments,
            "language": language
        }
        
        # Align the transcript to get word-level timestamps
        logger.info("Aligning transcript with audio for word-level timestamps")
        aligned_result = whisperx.align(
            whisperx_result["segments"],
            align_model,
            metadata,
            audio_path,
            device,
            return_char_alignments=False
        )
        
        # Now assign speakers to word segments
        logger.info("Assigning speakers to word segments")
        
        # Convert pyannote diarization format to whisperx format
        from pyannote.core import Annotation, Segment
        
        diarization = Annotation()
        for seg in speaker_segments:
            segment = Segment(seg["start"], seg["end"])
            diarization[segment] = seg["speaker"]
        
        # Assign speakers using WhisperX's assign_word_speakers
        result_with_speakers = whisperx.assign_word_speakers(
            diarization, 
            aligned_result
        )
        
        # Convert back to our format and aggregate by segment
        enhanced_segments = []
        
        for segment in result_with_speakers["segments"]:
            # Get the most common speaker in this segment based on word assignments
            word_speakers = []
            
            if "words" in segment:
                for word in segment["words"]:
                    if "speaker" in word:
                        word_speakers.append(word["speaker"])
            
            # Assign the most common speaker to the segment
            if word_speakers:
                from collections import Counter
                speaker_counts = Counter(word_speakers)
                assigned_speaker = speaker_counts.most_common(1)[0][0]
            else:
                # Fallback to overlap-based assignment for this segment
                assigned_speaker = self._get_speaker_by_overlap(
                    segment, 
                    speaker_segments
                )
            
            # Create enhanced segment
            enhanced_segment = segment.copy()
            enhanced_segment["speaker"] = assigned_speaker
            
            # Keep word-level speaker info if available
            if "words" in segment:
                enhanced_segment["words"] = segment["words"]
            
            enhanced_segments.append(enhanced_segment)
        
        logger.info(f"WhisperX alignment complete. Enhanced {len(enhanced_segments)} segments")
        return enhanced_segments

    def _assign_speakers_basic(
        self,
        transcript_segments: List[Dict[str, Any]],
        speaker_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Basic speaker assignment using segment-level overlap.
        
        This is the fallback method when WhisperX is not available.
        """
        enhanced_segments = []

        for segment in transcript_segments:
            assigned_speaker = self._get_speaker_by_overlap(segment, speaker_segments)
            
            # Create enhanced segment
            enhanced_segment = segment.copy()
            enhanced_segment["speaker"] = assigned_speaker or "SPEAKER_UNKNOWN"
            enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    def _get_speaker_by_overlap(
        self,
        segment: Dict[str, Any],
        speaker_segments: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Get the most likely speaker for a segment based on temporal overlap.
        """
        segment_start = segment.get("start", 0)
        segment_end = segment.get("end", 0)
        segment_mid = (segment_start + segment_end) / 2

        # Find the speaker who speaks the most during this segment
        speaker_overlaps = {}

        for speaker_seg in speaker_segments:
            speaker = speaker_seg["speaker"]
            spk_start = speaker_seg["start"]
            spk_end = speaker_seg["end"]

            # Calculate overlap duration
            overlap_start = max(segment_start, spk_start)
            overlap_end = min(segment_end, spk_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap_duration

        # Assign the speaker with maximum overlap
        if speaker_overlaps:
            return max(speaker_overlaps, key=speaker_overlaps.get)
        
        # Fallback: find speaker whose segment contains the midpoint
        for speaker_seg in speaker_segments:
            if speaker_seg["start"] <= segment_mid <= speaker_seg["end"]:
                return speaker_seg["speaker"]

        # If still no match, use the closest speaker segment
        if speaker_segments:
            closest_seg = min(
                speaker_segments,
                key=lambda s: min(
                    abs(s["start"] - segment_mid),
                    abs(s["end"] - segment_mid)
                )
            )
            return closest_seg["speaker"]
        
        return None

    def get_speaker_statistics(
        self,
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about speakers in the audio.

        Args:
            speaker_segments: Speaker diarization segments

        Returns:
            Dictionary with speaker statistics
        """
        if not speaker_segments:
            return {
                "num_speakers": 0,
                "total_speech_duration": 0,
                "speakers": {}
            }

        speakers = {}
        total_duration = 0

        for segment in speaker_segments:
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]
            total_duration += duration

            if speaker not in speakers:
                speakers[speaker] = {
                    "total_duration": 0,
                    "num_segments": 0,
                    "percentage": 0
                }

            speakers[speaker]["total_duration"] += duration
            speakers[speaker]["num_segments"] += 1

        # Calculate percentages
        for speaker_data in speakers.values():
            speaker_data["percentage"] = (
                speaker_data["total_duration"] / total_duration * 100
                if total_duration > 0 else 0
            )

        return {
            "num_speakers": len(speakers),
            "total_speech_duration": total_duration,
            "speakers": speakers
        }

    def format_transcript_with_speakers(
        self,
        enhanced_segments: List[Dict[str, Any]],
        include_timestamps: bool = True
    ) -> str:
        """
        Format transcript with speaker labels.

        Args:
            enhanced_segments: Transcript segments with speaker labels
            include_timestamps: Whether to include timestamps

        Returns:
            Formatted transcript string
        """
        lines = []
        current_speaker = None

        for segment in enhanced_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)

            if not text:
                continue

            # Add speaker label when speaker changes
            if speaker != current_speaker:
                if lines:  # Add blank line between speakers
                    lines.append("")

                if include_timestamps:
                    lines.append(f"[{speaker}] ({start:.2f}s - {end:.2f}s)")
                else:
                    lines.append(f"[{speaker}]")

                current_speaker = speaker

            # Add the text
            if include_timestamps:
                lines.append(f"  [{start:.2f}s] {text}")
            else:
                lines.append(f"  {text}")

        return "\n".join(lines)

    def format_transcript_for_ai_clips(
        self,
        enhanced_segments: List[Dict[str, Any]]
    ) -> str:
        """
        Format transcript in the optimal format for AI clip generation.
        
        Output format:
        start_time end_time SPEAKER_ID  Transcript text
        
        Example:
        0.522 2.609 SPEAKER_01  Sir, you there.
        2.609 5.338 SPEAKER_01 Are you going to teach your son to be a fighter?
        
        This format is:
        - Easy for AI to parse
        - Shows exact timing for clip extraction
        - Clearly identifies speakers
        - One line per segment for easy processing
        
        Args:
            enhanced_segments: Transcript segments with speaker labels
            
        Returns:
            Formatted transcript string optimized for AI processing
        """
        lines = []
        
        for segment in enhanced_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            if not text:
                continue
            
            # Format: start end SPEAKER text
            # Use fixed-width formatting for alignment
            line = f"{start:.3f} {end:.3f} {speaker}  {text}"
            lines.append(line)
        
        return "\n".join(lines)


# Global instance (lazy loaded)
_diarization_service: Optional[SpeakerDiarization] = None


def get_diarization_service(hf_token: Optional[str] = None) -> SpeakerDiarization:
    """Get or create the global speaker diarization service"""
    global _diarization_service

    if _diarization_service is None:
        _diarization_service = SpeakerDiarization(hf_token=hf_token)

    return _diarization_service


if __name__ == "__main__":
    """Test speaker diarization"""
    import sys

    # Test with an audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Testing speaker diarization on: {audio_file}")

        service = get_diarization_service()

        if service.is_available():
            # Run diarization
            segments = service.diarize(audio_file)

            if segments:
                print(f"\nFound {len(segments)} speaker segments:")
                for seg in segments[:10]:  # Show first 10
                    print(f"  {seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")

                # Get statistics
                stats = service.get_speaker_statistics(segments)
                print(f"\nSpeaker Statistics:")
                print(f"  Number of speakers: {stats['num_speakers']}")
                print(f"  Total speech duration: {stats['total_speech_duration']:.2f}s")
                for speaker, data in stats['speakers'].items():
                    print(f"  {speaker}: {data['total_duration']:.2f}s ({data['percentage']:.1f}%)")
        else:
            print("Speaker diarization not available. Check HF_TOKEN configuration.")
    else:
        print("Usage: python speaker_diarization.py <audio_file>")
        print("\nMake sure to:")
        print("1. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. Get token from: https://huggingface.co/settings/tokens")
        print("3. Set HF_TOKEN environment variable")
