from src.detectors.detector_licence_plates import DetectorLicencePlates
from src.experiments.detection_runner import run

if __name__ == '__main__':
    run(DetectorLicencePlates(debug=True))
