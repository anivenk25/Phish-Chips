import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure CPU usage only

import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import csv
import soundfile as sf

class OfficeAmbienceDetector:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_map = self._load_class_map()

        # Expanded + looser office-related tags
        self.office_tags = {
            'Typing': 1.0,
            'Computer keyboard': 1.0,
            'Speech': 0.6,
            'Conversation': 0.6,
            'Telephone': 0.4,
            'Air conditioning': 0.3,
            'White noise': 0.3,
            'Click': 0.3,
            'Printer': 0.3,
            'Office': 1.2
        }

        self.min_tag_threshold = 0.25   # Lowered detection threshold
        self.required_score = 0.1       # Final composite score cutoff (relaxed)
        self.strong_indicators = {'Typing', 'Computer keyboard', 'Office'}

    def _load_class_map(self):
        class_map_path = tf.keras.utils.get_file(
            'yamnet_class_map.csv',
            'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        )
        with open(class_map_path, newline='') as f:
            return [row['display_name'] for row in csv.DictReader(f)]

    def detect_office(self, audio_path):
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = waveform[:len(waveform) - (len(waveform) % 16000)]

        scores, embeddings, spectrogram = self.model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        detected_tags = []
        composite_score = 0.0
        used_weights = 0.0
        strong_hit = False

        for idx, score in enumerate(mean_scores):
            class_name = self.class_map[idx]
            if class_name in self.office_tags and score > self.min_tag_threshold:
                detected_tags.append(class_name)
                weight = self.office_tags[class_name]
                composite_score += score * weight
                used_weights += weight
                if class_name in self.strong_indicators:
                    strong_hit = True

        if used_weights > 0:
            composite_score /= used_weights

        # Show top 8 scores for debug
        print("\n🔍 Top Detected Tags:")
        for name, score in sorted([(self.class_map[i], float(s)) for i, s in enumerate(mean_scores)],
                                  key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {name}: {score:.2f}")

        return {
            'is_office': strong_hit or composite_score > self.required_score,
            'confidence': float(composite_score),
            'detected_tags': detected_tags,
            'composite_score': composite_score,
            'strong_signal': strong_hit
        }

def convert_to_wav_if_needed(path):
    if path.endswith('.ogg'):
        y, sr = librosa.load(path, sr=16000)
        wav_path = path.replace('.ogg', '.wav')
        sf.write(wav_path, y, sr)
        return wav_path
    return path

# === MAIN ===
if __name__ == "__main__":
    audio_file = convert_to_wav_if_needed("/home/r12/Downloads/WhatsApp Ptt 2025-06-20 at 1.54.58 AM.ogg")

    detector = OfficeAmbienceDetector()
    result = detector.detect_office(audio_file)

    print("\n=== 📊 Office Ambience Detection ===")
    if result['is_office']:
        print("✅ Office ambience detected")
    else:
        print("❌ No office ambience detected")
    print(f"Confidence score: {result['confidence']:.2%}")
    print(f"Detected tags: {', '.join(result['detected_tags']) if result['detected_tags'] else 'None'}")
