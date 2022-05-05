import time
from tabulate import tabulate
import cv2
from textAreaDetector import LicencePlatesParser1
from textAreaDetector2 import LicencePlatesParser2

PLATES_IMAGE_PATH = ['resources/plate1.jpg', 'resources/plate2.jpg', 'resources/plate3.jpg', 'resources/plate4.jpg',
                     'resources/plate5.jpg', 'resources/plate6.jpg', 'resources/plate7.jpg', 'resources/plate8.jpg',
                     'resources/plate9.jpg']
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
def detector1():
    parser = LicencePlatesParser2(False)
    detections = []
    for image in PLATES_IMAGE:
        text_result, debug_image = parser.run(image)
        detections.append(text_result)
    return detections


@measurement
def detector2():
    parser = LicencePlatesParser1()
    detections = []
    for image in PLATES_IMAGE:
        text_results = parser.run(image)
        detections.append(text_results)
    return detections


def test():
    result1, time1 = detector1()
    result2, time2 = detector2()
    table = [['detector1', 'detector2', 'real']]
    for idx, plates_text in enumerate(PLATES_TEXT):
        table.append([result1[idx], result2[idx], plates_text])
    table.append([round(time1, 2), round(time2, 2), "TIME MS"])
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=False))


if __name__ == '__main__':
    test()
