from src.detectors.detector_text_fields import DetectorTextFields
from src.experiments.detection_runner import run

if __name__ == '__main__':
    run(DetectorTextFields(debug=True))
