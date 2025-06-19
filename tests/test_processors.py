import unittest
from unittest.mock import patch, MagicMock, call
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import processors
from app.processors.ai_voice import process as ai_voice_process
from app.processors.ambience import process as ambience_process
from app.processors.diarization import process as diarization_process
from app.processors.emotion import process as emotion_process
from app.processors.transcription import process as transcription_process
from app.processors.scam_analysis import process as scam_analysis_process


class TestAIVoiceProcessor(unittest.TestCase):
    @patch('app.processors.ai_voice.Extractor')
    @patch('app.processors.ai_voice.torch.cuda.is_available', return_value=False)
    def test_process_cpu(self, mock_cuda_available, mock_extractor_class):
        # Setup fake extractor returning a fake detection
        fake_extractor = MagicMock()
        fake_extractor.detect_fake.return_value = (True, 0.98765)
        mock_extractor_class.return_value = fake_extractor
        result = ai_voice_process("dummy.wav", [])
        self.assertTrue(result['is_ai_voice'])
        self.assertAlmostEqual(result['ai_score'], round(0.98765, 4))
        mock_extractor_class.assert_called_once_with(
            encoder_model="damo/speech_personal_model",
            use_gpu=False
        )

    @patch('app.processors.ai_voice.Extractor')
    @patch('app.processors.ai_voice.torch.cuda.is_available', return_value=True)
    def test_process_gpu_fallback(self, mock_cuda_available, mock_extractor_class):
        # Simulate GPU OOM on first attempt, then success on CPU fallback
        fake_extractor_gpu = MagicMock()
        fake_extractor_gpu.detect_fake.side_effect = RuntimeError("CUDA out of memory")
        fake_extractor_cpu = MagicMock()
        fake_extractor_cpu.detect_fake.return_value = (False, 0.12345)
        mock_extractor_class.side_effect = [fake_extractor_gpu, fake_extractor_cpu]
        result = ai_voice_process("dummy.wav", [])
        self.assertFalse(result['is_ai_voice'])
        self.assertAlmostEqual(result['ai_score'], round(0.12345, 4))
        expected_calls = [
            call(encoder_model="damo/speech_personal_model", use_gpu=True),
            call(encoder_model="damo/speech_personal_model", use_gpu=False)
        ]
        mock_extractor_class.assert_has_calls(expected_calls)


class TestAmbienceProcessor(unittest.TestCase):
    @patch('app.processors.ambience.OfficeAmbienceDetector')
    def test_process_returns_detector_output(self, mock_detector_class):
        fake_detector = MagicMock()
        fake_detector.detect_office.return_value = {
            'is_office': True,
            'confidence': 0.8,
            'detected_tags': ['Typing'],
            'composite_score': 0.4
        }
        mock_detector_class.return_value = fake_detector
        result = ambience_process("dummy.wav")
        self.assertTrue(result['is_office'])
        self.assertEqual(result['confidence'], 0.8)
        self.assertListEqual(result['detected_tags'], ['Typing'])
        self.assertEqual(result['composite_score'], 0.4)


class TestTranscriptionProcessor(unittest.TestCase):
    @patch('app.processors.transcription.whisper.load_model')
    def test_process_returns_transcript_and_segments(self, mock_load_model):
        fake_model = MagicMock()
        fake_model.transcribe.return_value = {
            'text': 'Hello world',
            'segments': [{'id': 1, 'start': 0, 'end': 1, 'text': 'Hello world'}]
        }
        mock_load_model.return_value = fake_model
        result = transcription_process("dummy.wav", [])
        self.assertEqual(result['transcript'], 'Hello world')
        self.assertEqual(result['segments'], [{'id': 1, 'start': 0, 'end': 1, 'text': 'Hello world'}])
        mock_load_model.assert_called_once_with("medium", device="cpu")


class TestDiarizationProcessor(unittest.TestCase):
    @patch('app.processors.diarization.Pipeline')
    @patch('app.processors.diarization.torch.cuda.is_available', return_value=False)
    def test_process_returns_segments(self, mock_cuda_available, mock_pipeline_class):
        # Prepare fake diarization pipeline
        class FakeTurn:
            def __init__(self, start, end, duration):
                self.start = start
                self.end = end
                self.duration = duration

        class FakeDiar:
            def __init__(self, tracks):
                self._tracks = tracks
            def itertracks(self, yield_label=True):
                for turn, _, speaker in self._tracks:
                    yield turn, None, speaker

        class FakePipeline:
            def __init__(self, diar):
                self._diar = diar
            def to(self, device):
                return self
            def __call__(self, file_path):
                return self._diar

        # Create fake tracks
        turn1 = FakeTurn(0.0, 1.0, 1.0)
        turn2 = FakeTurn(1.0, 2.5, 1.5)
        fake_diar = FakeDiar([(turn1, None, 'Speaker1'), (turn2, None, 'Speaker2')])
        fake_pipeline = FakePipeline(fake_diar)
        mock_pipeline_class.from_pretrained.return_value = fake_pipeline
        result = diarization_process("dummy.wav", [])
        expected = [
            {'speaker': 'Speaker1', 'start': 0.0, 'end': 1.0, 'duration': 1.0},
            {'speaker': 'Speaker2', 'start': 1.0, 'end': 2.5, 'duration': 1.5}
        ]
        self.assertEqual(result, expected)


class TestEmotionProcessor(unittest.TestCase):
    @patch('app.processors.emotion.AutoProcessor')
    @patch('app.processors.emotion.AutoModelForAudioClassification')
    @patch('app.processors.emotion.torchaudio.load')
    def test_process_returns_top_emotion_and_probs(self, mock_torchaudio_load, mock_model_class, mock_processor_class):
        # Fake waveform and sample rate
        waveform = torch.tensor([[0.1, 0.2]])
        sr = 16000
        mock_torchaudio_load.return_value = (waveform, sr)

        # Fake processor
        class DummyProcessor:
            def __call__(self, audio, sampling_rate, return_tensors):
                return {'input_values': torch.tensor([[0.1, 0.2]])}

        dummy_processor = DummyProcessor()
        mock_processor_class.from_pretrained.return_value = dummy_processor

        # Fake model with logits and id2label
        id2label = {0: 'happy', 1: 'sad', 2: 'angry'}
        fake_model = MagicMock()
        fake_model.config.id2label = id2label
        fake_model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.7, 0.2]]))
        mock_model_class.from_pretrained.return_value = fake_model

        result = emotion_process("dummy.wav", [])
        self.assertEqual(len(result), 1)
        top = result[0]
        self.assertEqual(top['top_emotion'], 'sad')
        # Check probabilities keys
        self.assertSetEqual(set(top['emotion_probs'].keys()), set(id2label.values()))


class TestScamAnalysisProcessor(unittest.TestCase):
    @patch('app.processors.scam_analysis.run')
    def test_process_parses_valid_json(self, mock_run):
        fake_stdout = 'prefix\n{"is_scam": true, "confidence": 90, "red_flags": ["urgency"], "target": ["money"], "analysis": "test"}\nsuffix'
        mock_run.return_value = MagicMock(stdout=fake_stdout)
        transcription = {'transcript': 'dummy'}
        ai_voice = {'is_ai_voice': True, 'ai_score': 0.5}
        emotion = [{'top_emotion': 'happy'}]
        result = scam_analysis_process(transcription, ai_voice, emotion)
        self.assertTrue(result['is_scam'])
        self.assertEqual(result['confidence'], 90)
        self.assertListEqual(result['red_flags'], ['urgency'])
        self.assertListEqual(result['target'], ['money'])
        self.assertEqual(result['analysis'], 'test')

    @patch('app.processors.scam_analysis.subprocess.run')
    def test_process_handles_invalid_json(self, mock_run):
        mock_run.return_value = MagicMock(stdout='no json here')
        result = scam_analysis_process({}, {}, [])
        self.assertEqual(result, {
            'is_scam': False,
            'confidence': 0,
            'red_flags': [],
            'target': [],
            'analysis': ''
        })


if __name__ == '__main__':
    unittest.main()
