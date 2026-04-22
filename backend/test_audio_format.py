#!/usr/bin/env python3
"""
Test script to check if VoIP recordings are raw PCM data
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path

def test_raw_audio_conversion():
    """Test converting raw PCM data to proper WAV format"""
    
    recordings_dir = "/home/antonyshane/voip/signaling_server/recordings"
    test_file = None
    
    # Find first .wav file
    for file_name in os.listdir(recordings_dir):
        if file_name.endswith('.wav'):
            test_file = os.path.join(recordings_dir, file_name)
            break
    
    if not test_file:
        print("No .wav files found")
        return
    
    print(f"Testing file: {Path(test_file).name}")
    
    # Try different sample rates and formats
    sample_rates = [8000, 16000, 44100, 48000]
    dtypes = [np.int16, np.int32, np.float32]
    
    for sr in sample_rates:
        for dtype in dtypes:
            try:
                # Read as raw binary data
                with open(test_file, 'rb') as f:
                    raw_data = f.read()
                
                # Convert to numpy array
                audio_data = np.frombuffer(raw_data, dtype=dtype)
                
                # Check if data looks reasonable
                if len(audio_data) > 1000:  # At least 1000 samples
                    # Check for non-zero data
                    non_zero_count = np.count_nonzero(audio_data)
                    if non_zero_count > len(audio_data) * 0.1:  # At least 10% non-zero
                        print(f"  Potential match: {sr}Hz, {dtype}")
                        print(f"    Length: {len(audio_data)} samples ({len(audio_data)/sr:.2f}s)")
                        print(f"    Non-zero samples: {non_zero_count}/{len(audio_data)} ({100*non_zero_count/len(audio_data):.1f}%)")
                        print(f"    Value range: {audio_data.min()} to {audio_data.max()}")
                        
                        # Try to save as test WAV
                        test_output = f"test_conversion_{sr}_{dtype.__name__}.wav"
                        try:
                            # Normalize if needed
                            if dtype == np.float32:
                                normalized_data = audio_data
                            else:
                                normalized_data = audio_data.astype(np.float32) / np.iinfo(dtype).max
                            
                            sf.write(test_output, normalized_data, sr)
                            print(f"    ✅ Saved test file: {test_output}")
                        except Exception as e:
                            print(f"    ❌ Save failed: {e}")
                        print()
            except Exception as e:
                continue

if __name__ == "__main__":
    test_raw_audio_conversion()
