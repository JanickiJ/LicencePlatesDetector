import time
from tabulate import tabulate
import cv2
from src.detectors.detector_text_fields import DetectorTextFields
from src.detectors.detector_licence_plates import DetectorLicencePlates

PLATES_IMAGE_PATH = map(lambda source: '../../resources/' + source,
                        ['plate1.jpg', 'plate2.jpg', 'plate3.jpg', 'plate4.jpg',
                         'plate5.jpg', 'plate6.jpg', 'plate7.jpg', 'plate8.jpg',
                         'plate9.jpg'])
PLATES_TEXT = ['PO4PJ54', 'PO6SU70', 'DL4929E', 'PO8MH80', 'PO4PH36', 'PZ4444M', 'WE286YU', 'TOSZTOS', 'S2YBKI']
PLATES_IMAGE = [cv2.imread(path) for path in PLATES_IMAGE_PATH]


def measurement(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        detections = f(*args, **kwargs)
        time2 = time.time()
        time_diff = (time2 - time1) * 1000.0
        return detections, time_diff

    return wrap


@measurement
def detection(detector):
    detections = []
    for image in PLATES_IMAGE:
        text_result, debug_image = detector.run(image)
        detections.append(text_result)
    return detections


def test():
    result1, time1 = detection(DetectorLicencePlates(False))
    result2, time2 = detection(DetectorTextFields(False))

    table = [['detector1', 'detector2', 'real']]
    for idx, plates_text in enumerate(PLATES_TEXT):
        table.append([result1[idx], result2[idx], plates_text])
    table.append([round(time1, 2), round(time2, 2), "TIME MS"])
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=False))


if __name__ == '__main__':
    test()
