#!/bin/bash

echo "🧪 Testing VoIP Recording Upload"
echo "================================"

# Create a test audio file (silent WAV)
echo "📝 Creating test audio file..."
ffmpeg -f lavfi -i anullsrc=channel_layout=mono:sample_rate=44100 -t 5 test_recording.wav -y

# Check if file was created
if [ -f "test_recording.wav" ]; then
    echo "✅ Test audio file created: test_recording.wav"
    
    # Upload to server
    echo "📤 Uploading to VoIP server..."
    curl -X POST \
         -F "recording=@test_recording.wav" \
         -F "timestamp=$(date -Iseconds)" \
         -F "type=test_recording" \
         -F "analysis_priority=scam_detection" \
         -F "chunk_interval=10_seconds" \
         -F "call_duration=5000" \
         http://localhost:3000/upload-recording
    
    echo ""
    echo "📁 Checking recordings directory..."
    ls -la /home/antonyshane/voip/signaling_server/recordings/
    
    # Cleanup
    rm test_recording.wav
else
    echo "❌ Failed to create test audio file"
    echo "Install ffmpeg: sudo apt install ffmpeg"
fi
