"""
Quick test for speaker diarization using existing config.
This test uses the config.py file which loads .env automatically.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("\n" + "="*80)
    print(" "*25 + "DIARIZATION QUICK TEST")
    print("="*80)
    
    # Step 1: Check config
    print("\n1. Loading configuration...")
    try:
        from config import settings
        print(f"✅ Config loaded")
        print(f"   HF_TOKEN: {settings.HF_TOKEN[:10]}..." if settings.HF_TOKEN else "   ❌ HF_TOKEN not found")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    if not settings.HF_TOKEN:
        print("\n❌ HF_TOKEN not configured. Please add to .env file.")
        return
    
    # Step 2: Test PyAnnote access
    print("\n2. Testing PyAnnote model access...")
    try:
        from pyannote.audio import Pipeline
        
        print("   Loading speaker diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.HF_TOKEN
        )
        print(f"✅ PyAnnote pipeline loaded successfully!")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to load pipeline: {error_msg[:200]}")
        
        if "403" in error_msg:
            print("\n⚠️  ACCESS DENIED - You need to:")
            print("   1. Go to: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   2. Click 'Agree and access repository'")
            print("   3. Go to: https://huggingface.co/pyannote/segmentation-3.0")
            print("   4. Click 'Agree and access repository'")
            print("   5. Wait 2-5 minutes, then run this test again")
            return
        else:
            print(f"\n⚠️  Error: {error_msg}")
            return
    
    # Step 3: Get test file
    print("\n3. Provide a test file...")
    print("   Enter path to a video/audio file with multiple speakers:")
    print("   (Or press Enter to skip transcription test)")
    
    file_path = input("\n   File path: ").strip().strip('"')
    
    if not file_path:
        print("\n✅ Configuration test complete! Diarization should work in backend.")
        return
    
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        return
    
    # Step 4: Test transcription
    print(f"\n4. Testing transcription with diarization...")
    print(f"   File: {file_path}")
    print("   ⏳ This may take 2-5 minutes...")
    
    try:
        from services.transcription import transcribe_audio_sync
        
        result = transcribe_audio_sync(
            file_path,
            model_size="tiny",
            enable_diarization=True,
            min_speakers=2,
            max_speakers=5
        )
        
        if result and result.get("segments"):
            print(f"\n✅ Transcription completed!")
            print(f"   Total segments: {len(result['segments'])}")
            
            if result.get("speaker_segments"):
                print(f"\n✅ SPEAKER DIARIZATION SUCCESSFUL!")
                print(f"   Speakers detected: {result['speaker_stats']['num_speakers']}")
                
                for speaker, stats in result['speaker_stats']['speakers'].items():
                    print(f"   - {speaker}: {stats['duration']:.1f}s ({stats['percentage']:.1f}%)")
                
                # Save transcript
                if result.get("transcript_for_ai"):
                    output_file = "test_transcript_output.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(result["transcript_for_ai"])
                    
                    print(f"\n   Transcript preview (first 10 lines):")
                    for line in result['transcript_for_ai'].split('\n')[:10]:
                        if line.strip():
                            print(f"   {line}")
                    
                    print(f"\n✅ Full transcript saved to: {output_file}")
                    print("\n🎉 ALL TESTS PASSED! Speaker diarization is working!")
            else:
                print("\n⚠️  Transcription succeeded but no speakers detected")
                print("   This might be normal for single-speaker content")
        else:
            print("\n❌ Transcription failed")
            
    except Exception as e:
        print(f"\n❌ Transcription error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
