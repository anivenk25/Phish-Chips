import librosa
from panns_inference import AudioTagging

# Tags and weights used to compute office ambience score
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
    def __init__(self, threshold: float = 0.6):
        # Initialize PANNs audio tagging model on CPU
        self.model = AudioTagging(checkpoint_path=None, device='cpu')
        self.threshold = threshold

    def detect_office(self, audio_path: str) -> dict:
        """Detect office ambience in audio file"""
        # Load and resample audio to 32 kHz mono
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        # Add batch dimension
        batch = audio[None, :]
        # Perform inference
        clipwise_output, _ = self.model.inference(batch)
        # Remove batch dim
        scores = clipwise_output[0]

        # Get class labels from PANNs model config
        classes = self.model.labels

        # Compute composite score and detect tags
        office_score = 0.0
        detected_tags = []
        for idx, score in enumerate(scores):
            class_name = classes[idx]
            if class_name in OFFICE_TAGS:
                weight = OFFICE_TAGS[class_name]
                if score > weight:
                    detected_tags.append(class_name)
                office_score += score * weight
        # Normalize
        if OFFICE_TAGS:
            office_score /= sum(OFFICE_TAGS.values())

        return {
            'is_office': office_score > self.threshold,
            'confidence': float(office_score),
            'detected_tags': detected_tags,
            'composite_score': office_score
        }

def process(file_path: str) -> dict:
    """
    Detects office ambience sounds using PANNs model.
    """
    detector = OfficeAmbienceDetector()
    return detector.detect_office(file_path)
