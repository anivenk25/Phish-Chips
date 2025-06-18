import os
import pandas as pd
import subprocess

OPENSMILE_PATH = "/usr/local/bin/SMILExtract"  # Change if installed elsewhere
CONFIG_PATH = "/home/yourname/opensmile/config/emobase/emobase.conf"  # Update path
TEMP_CSV = "output.csv"

def detect_emotion(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    print("🔁 Running openSMILE for emotion feature extraction...")

    # Run openSMILE to extract features
    command = [
        OPENSMILE_PATH,
        "-C", CONFIG_PATH,
        "-I", audio_path,
        "-O", TEMP_CSV
    ]
    
    subprocess.run(command, check=True)

    print("📈 Reading extracted features...")
    df = pd.read_csv(TEMP_CSV, delimiter=';', skiprows=1)
    
    # Basic emotion estimation (very naive)
    # You can replace this logic with a better trained classifier
    arousal = df["F0env_sma"].mean()
    pitch = df["F0_sma"].mean()

    print("\n📊 Emotion estimate based on pitch/arousal:")
    if pitch > 200 and arousal > 0.05:
        emotion = "Excited or Happy"
    elif pitch < 140 and arousal < 0.01:
        emotion = "Calm or Sad"
    elif arousal > 0.1:
        emotion = "Angry or Anxious"
    else:
        emotion = "Neutral"

    print(f"🧠 Detected Emotion: {emotion}")

if __name__ == "__main__":
    audio_file = "/home/r12/Downloads/WhatsApp Ptt 2025-06-17 at 5.26.59 PM.ogg"  # Update to your file
    detect_emotion(audio_file)

