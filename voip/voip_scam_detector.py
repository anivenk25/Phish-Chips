#!/usr/bin/env python3
"""
VoIP Scam Detection Integration Script
Automatically processes VoIP call recordings for scam detection
"""

import os
import sys
import time
import json
import shutil
import asyncio
import requests
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
VOIP_RECORDINGS_DIR = "./recordings"  # Local recordings directory
SCAM_API_URL = "http://localhost:3000"  # Your scam detection API
PROCESSED_DIR = "./processed_recordings"
RESULTS_DIR = "./analysis_results"

# Create directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class VoIPScamDetector(FileSystemEventHandler):
    """Monitors VoIP recordings and sends them for scam analysis"""
    
    def __init__(self):
        self.processed_files = set()
        print("🎯 VoIP Scam Detector initialized")
        print(f"📁 Monitoring: {os.path.abspath(VOIP_RECORDINGS_DIR)}")
        print(f"🔗 API: {SCAM_API_URL}")
    
    def on_created(self, event):
        if event.is_dir:
            return
        
        file_path = event.src_path
        if self.is_audio_file(file_path):
            print(f"🎤 New VoIP recording: {Path(file_path).name}")
            # Wait for file to be fully written
            time.sleep(2)
            self.analyze_recording(file_path)
    
    def is_audio_file(self, file_path):
        """Check if file is an audio file"""
        audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
        return Path(file_path).suffix.lower() in audio_extensions
    
    def analyze_recording(self, file_path):
        """Send recording to scam detection API"""
        if file_path in self.processed_files:
            return
        
        try:
            print(f"🔍 Analyzing: {Path(file_path).name}")
            
            # Send to scam detection API
            with open(file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                response = requests.post(
                    f"{SCAM_API_URL}/analyze-audio",
                    files=files,
                    timeout=120  # 2 minute timeout for analysis
                )
            
            if response.status_code == 200:
                result = response.json()
                self.handle_analysis_result(file_path, result)
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Cannot connect to scam detection API at {SCAM_API_URL}")
            print("Make sure the scam detection server is running!")
        except Exception as e:
            print(f"❌ Error analyzing {Path(file_path).name}: {e}")
        
        self.processed_files.add(file_path)
    
    def handle_analysis_result(self, file_path, result):
        """Handle scam analysis results"""
        filename = Path(file_path).name
        
        # Save detailed results
        result_file = os.path.join(RESULTS_DIR, f"analysis_{Path(file_path).stem}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Display results
        is_scam = result.get('scam', False)
        confidence = result.get('confidence_score', 0.0)
        reasoning = result.get('reasoning', 'No reasoning provided')
        
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS COMPLETE: {filename}")
        print(f"{'='*60}")
        
        if is_scam:
            print(f"🚨 SCAM DETECTED! (Confidence: {confidence:.2f})")
            print(f"📋 Reason: {reasoning}")
            
            # Additional scam indicators
            if result.get('is_cloned'):
                print(f"🎭 Voice cloning detected (confidence: {result.get('clone_confidence', 0):.2f})")
            
            emotions = result.get('emotions', [])
            if emotions:
                print(f"😠 Emotions detected: {[e.get('top_emotion') for e in emotions]}")
            
            transcript = result.get('transcript', '')
            if transcript:
                print(f"💬 Transcript: {transcript[:200]}...")
                
        else:
            print(f"✅ Clean call (Confidence: {confidence:.2f})")
            print(f"📋 Analysis: {reasoning}")
        
        print(f"{'='*60}\n")
        
        # Move processed file
        processed_path = os.path.join(PROCESSED_DIR, filename)
        shutil.move(file_path, processed_path)
        print(f"📦 Moved to processed: {filename}")

def check_api_health():
    """Check if scam detection API is running"""
    try:
        response = requests.get(f"{SCAM_API_URL}/voip-status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Scam Detection API is running")
            print(f"📊 Status: {status.get('status')}")
            return True
    except:
        pass
    
    print(f"❌ Scam Detection API not available at {SCAM_API_URL}")
    print("Please start the scam detection server first:")
    print("cd /path/to/scam_detection && python main.py")
    return False

def main():
    """Main monitoring loop"""
    print("🚀 VoIP Scam Detection Integration Starting...")
    
    # Check if recordings directory exists
    if not os.path.exists(VOIP_RECORDINGS_DIR):
        print(f"❌ Recordings directory not found: {VOIP_RECORDINGS_DIR}")
        print("Make sure you're running this from the VoIP signaling server directory!")
        return
    
    # Check API health
    if not check_api_health():
        print("⚠️ Continuing anyway - will retry when files are detected")
    
    # Set up file monitoring
    event_handler = VoIPScamDetector()
    observer = Observer()
    observer.schedule(event_handler, VOIP_RECORDINGS_DIR, recursive=False)
    observer.start()
    
    print("👁️ Monitoring VoIP recordings for scam detection...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping VoIP scam detection...")
        observer.stop()
    
    observer.join()
    print("✅ VoIP scam detection stopped")

if __name__ == "__main__":
    main()
