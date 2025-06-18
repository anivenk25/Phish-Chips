from background_noise import OfficeAmbienceDetector

def process(file_path: str) -> dict:
    """
    Detects office ambience sounds using PANNs model.
    """
    detector = OfficeAmbienceDetector()
    return detector.detect_office(file_path)
