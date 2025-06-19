import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class Extractor:
    """
    Deepfake audio detection using Hugging Face transformers model.
    """
    def __init__(self, encoder_model: str, use_gpu: bool):
        # Use GPU if specified
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.threshold = 0.5
        # Load deepfake detection model using specified encoder_model
        model_name = encoder_model
        # Support private Hugging Face repos by passing token via environment
        import os
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        try:
            if hf_token:
                self.model = AutoModelForAudioClassification.from_pretrained(
                    model_name,
                    weights_only=True,
                    use_auth_token=hf_token,
                )
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name,
                    use_auth_token=hf_token,
                )
            else:
                self.model = AutoModelForAudioClassification.from_pretrained(
                    model_name,
                    weights_only=True,
                )
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                "Ensure the repo exists and you have access. "
                "You may need to set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable, "
                f"original error: {e}"
            ) from e

    def detect_fake(self, file_path: str):
        # Load audio
        wav, sr = torchaudio.load(file_path)
        # Resample if needed
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        # Prepare numpy array for feature extractor
        audio = wav.squeeze().numpy()
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        # Move inputs to correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        # Index 1 is fake
        fake_prob = probs[1].item()
        is_fake = fake_prob > self.threshold
        return is_fake, fake_prob