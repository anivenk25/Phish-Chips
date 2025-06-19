from panns_inference import AudioTagging, SoundEventDetection
import librosa
import numpy as np

OFFICE_TAGS = {
    'Computer keyboard': 0.4,
    'Typing': 0.5,
    'Printer': 0.3,
    'Chatter': 0.6,
    'Telephone': 0.4,
    'Office': 0.7,
    'White noise': 0.5,
    'Air conditioning': 0.4,
    'Click': 0.3,
    'Writing': 0.3
}

class OfficeAmbienceDetector:
    def __init__(self, threshold=0.6):
        self.model = AudioTagging(checkpoint_path=None, device='cpu')
        self.threshold = threshold
        
    def detect_office(self, audio_path):
        """Detect office ambience in audio file"""
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=32000)
        
        # Make batch dimension
        audio = audio[None, :]
        
        # Inference
        clipwise_output, _ = self.model.inference(audio)
        clipwise_output = clipwise_output[0]  # Remove batch dim
        
        # Get class labels from PANNs model config
        classes = self.model.labels
        
        # Analyze results
        detections = []
        office_score = 0
        detected_tags = []
        
        for idx, score in enumerate(clipwise_output):
            class_name = classes[idx]
            if class_name in OFFICE_TAGS:
                if score > OFFICE_TAGS[class_name]:
                    detected_tags.append(class_name)
                office_score += score * OFFICE_TAGS[class_name]
        
        # Normalize score
        if OFFICE_TAGS:
            office_score /= sum(OFFICE_TAGS.values())
        
        return {
            'is_office': office_score > self.threshold,
            'confidence': float(office_score),
            'detected_tags': detected_tags,
            'composite_score': office_score
        }

# Usage example
if __name__ == "__main__":
    detector = OfficeAmbienceDetector()
    result = detector.detect_office("office_recording.wav")
    print("Office ambience detected:" if result['is_office'] else "No office detected")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Detected sounds: {', '.join(result['detected_tags'])}")