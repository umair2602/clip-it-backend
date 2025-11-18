"""
Example: Generate AI-ready transcript with speaker diarization.

This script demonstrates how to get a transcript in the format:
start_time end_time SPEAKER_ID  Text

Which is perfect for feeding to AI for clip generation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.transcription import transcribe_audio


async def generate_ai_transcript(video_path: str):
    """
    Generate a transcript optimized for AI clip generation.
    
    Output format:
    0.522 2.609 SPEAKER_01  Sir, you there.
    2.609 5.338 SPEAKER_01 Are you going to teach your son to be a fighter?
    5.338 5.98 SPEAKER_01 Bring him up to be a fighter?
    6.02 7.845 SPEAKER_00  No, sir.
    """
    
    print("="*70)
    print("Generating AI-Ready Transcript with Speaker Diarization")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print("\nProcessing...")
    
    # Transcribe with speaker diarization enabled
    result = await transcribe_audio(
        video_path=video_path,
        model_size="base",  # Use 'base' for good accuracy
        enable_diarization=True,
        # Optional: specify if you know the number of speakers
        # num_speakers=2  # For interviews
        # min_speakers=2, max_speakers=4  # For panels
    )
    
    # Get the AI-optimized transcript
    ai_transcript = result.get("transcript_for_ai")
    
    if ai_transcript:
        print("\n" + "="*70)
        print("AI-READY TRANSCRIPT (Copy this to give to AI)")
        print("="*70)
        print(ai_transcript)
        print("="*70)
        
        # Also show speaker statistics
        stats = result.get("speaker_stats")
        if stats:
            print(f"\n📊 Found {stats['num_speakers']} speakers:")
            for speaker, data in stats['speakers'].items():
                print(f"  {speaker}: {data['total_duration']:.1f}s ({data['percentage']:.1f}%)")
        
        # Save to file
        output_file = Path(video_path).stem + "_transcript_for_ai.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ai_transcript)
        
        print(f"\n💾 Saved to: {output_file}")
        
        # Also show comparison with human-readable format
        readable_transcript = result.get("text_with_speakers")
        if readable_transcript:
            readable_file = Path(video_path).stem + "_transcript_readable.txt"
            with open(readable_file, "w", encoding="utf-8") as f:
                f.write(readable_transcript)
            print(f"💾 Readable version saved to: {readable_file}")
        
        print("\n✅ Done! You can now feed the AI-ready transcript to your AI model.")
        print("\nExample prompt for AI:")
        print("-" * 70)
        print("""
Generate viral clips from this transcript. Each line shows:
start_time end_time SPEAKER_ID Text

Find moments that are:
- Emotional or impactful
- Controversial or thought-provoking  
- Funny or entertaining
- Educational with key insights

For each clip, provide:
1. Start and end times
2. Why it would make a good clip
3. Suggested title

Transcript:
[paste the transcript here]
        """)
        print("-" * 70)
        
    else:
        print("\n❌ No transcript generated. Check that speaker diarization is enabled.")
    
    return result


async def main():
    if len(sys.argv) < 2:
        print("Usage: python example_ai_transcript.py <video_file>")
        print("\nExample:")
        print("  python example_ai_transcript.py path/to/interview.mp4")
        print("\nThis will generate a transcript in the format:")
        print("  0.522 2.609 SPEAKER_01  Sir, you there.")
        print("  2.609 5.338 SPEAKER_01 Are you going to teach your son to be a fighter?")
        print("\nPerfect for feeding to AI for clip generation!")
        return
    
    video_path = sys.argv[1]
    
    try:
        await generate_ai_transcript(video_path)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
